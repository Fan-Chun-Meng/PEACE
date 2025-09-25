import numpy as np
import torch
import random
from torch_geometric.data import DataLoader, Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lib.utils as utils
from torch.nn.utils.rnn import pad_sequence
import logging
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class ParseData(object):

    def __init__(self, dataset_path, args, suffix='_springs5', mode="extrap", domain_sampling_strategy='balanced', id_ood_ratio=0.7, enable_normalization=True):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.extrap_num = args.extrap_num
        # 初始化ODE相关参数
        self.total_ode_step = args.total_ode_step
        self.total_step_train = args.total_ode_step_train
        self.num_pre = args.extrap_num
        self.domain_num = args.num_domains
        self.num_condition = args.condition_num
        self.cutting_edge = args.cutting_edge
        
        print("num_condition:{}, num_predict:{}, total_step:{}".format(
            self.num_condition, self.num_pre, self.total_ode_step))
        
        # 初始化域相关属性
        self.domain_sampling_strategy = domain_sampling_strategy
        self.id_ood_ratio = id_ood_ratio
        self.current_domain_id = 0  # 默认为ID域
        
        # 初始化域统计信息存储
        self.domain_stats = {}
        self.domain_sample_counts = {}
        self.domain_param_ranges = {}
        
        # 初始化域数据加载器和路径
        self.domain_data_loaders = {}
        self.domain_batch_sizes = {}
        self.domain_data_paths = {}
        
        # 初始化全局标准化参数
        self.global_norm_params = None
        self.enable_normalization = enable_normalization  # 控制是否启用归一化

        # 标准化参数
        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None

        # 设置随机种子
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        
    def _compute_domain_stats(self, loc, vel, sys_paras, domain_id):
        """计算域统计信息
        Args:
            loc: 位置数据 [num_atoms, timesteps, 2]
            vel: 速度数据 [num_atoms, timesteps, 2]
            sys_paras: 系统参数 [4]
            domain_id: 域ID
        Returns:
            dict: 包含样本数量和参数范围的统计信息
        """
        stats = {
            'param_range': {
                'box_size': float(sys_paras[0]),
                'vel_norm': float(sys_paras[1]), 
                'interaction_strength': float(sys_paras[2]),
                'spring_prob': float(sys_paras[3])
            },
            'feature_stats': {
                'loc': {
                    'mean': float(np.mean(loc)),
                    'std': float(np.std(loc)), 
                    'min': float(np.min(loc)),
                    'max': float(np.max(loc))
                },
                'vel': {
                    'mean': float(np.mean(vel)),
                    'std': float(np.std(vel)),
                    'min': float(np.min(vel)), 
                    'max': float(np.max(vel))
                }
            },
            'is_id_domain': domain_id == 0
        }

        # 更新域统计信息
        if not hasattr(self, 'domain_stats'):
            self.domain_stats = {}
        if not hasattr(self, 'domain_sample_counts'):
            self.domain_sample_counts = {}
        if not hasattr(self, 'domain_param_ranges'):
            self.domain_param_ranges = {}
            
        self.domain_stats[domain_id] = stats
        self.domain_param_ranges[domain_id] = stats['param_range']
        
        # 记录日志
        logger.info(f"域 {domain_id} 统计信息:")
        logger.info(f"  参数范围: {stats['param_range']}")
        logger.info(f"  特征统计: {stats['feature_stats']}")
        
        return stats

        # 多域相关属性
        self.domain_sampling_strategy = domain_sampling_strategy  # 域采样策略：balanced, id_focus, domain_aware
        self.id_ood_ratio = id_ood_ratio  # ID数据在批次中的比例
        self.domain_stats = {}  # 域统计信息
        self.current_domain_id = None  # 当前处理的域ID
        self.domain_id = None  # 当前数据的域ID
        self.domain_data_paths = {}  # 存储不同域的数据路径
        self.domain_data_loaders = {}  # 存储不同域的数据加载器
        self.domain_batch_sizes = {}  # 存储不同域的批次大小
        

        
        # 全局标准化参数（用于多域）
        self.global_norm_params = None
        
        # 域数据统计
        self.domain_sample_counts = {}  # 每个域的样本数量
        self.domain_param_ranges = {}  # 每个域的参数范围

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


    def load_data(self, sample_percent, batch_size, data_type="train", domain_id=0, ood_data_path=None):
        """加载数据，支持多域数据加载
        
        Args:
            sample_percent: 采样百分比
            batch_size: 批次大小
            data_type: 数据类型，'train'或'test'
            domain_id: 域ID，0表示ID域，其他值表示OOD域
            ood_data_path: OOD数据路径，当domain_id不为0时需要提供
        """
        self.batch_size = batch_size
        self.sample_percent = sample_percent
        self.current_domain_id = domain_id
        
        if data_type == "train":
            cut_num = 20000
        else:
            cut_num = 5000

        # 根据域ID选择数据路径
        data_path = self.dataset_path if domain_id == 0 else ood_data_path
        if data_path is None and domain_id > 0:
            raise ValueError(f"OOD data path is required for domain_id {domain_id}")
        
        # 存储域数据路径和批次大小
        self.domain_data_paths[domain_id] = data_path
        self.domain_batch_sizes[domain_id] = batch_size

        # Loading Data with domain tracking
        loc = np.load(data_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        vel = np.load(data_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        edges = np.load(data_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # [500,5,5]
        times = np.load(data_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # [500，5]
        sys_paras = np.load(data_path + '/paras_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # [500,4]

        # 更新域统计信息
        self.domain_stats[domain_id] = {
            'samples': len(loc),
            'param_range': {
                'box_size': [float(np.min(sys_paras[:, 0])), float(np.max(sys_paras[:, 0]))],
                'vel_norm': [float(np.min(sys_paras[:, 1])), float(np.max(sys_paras[:, 1]))],
                'interaction_strength': [float(np.min(sys_paras[:, 2])), float(np.max(sys_paras[:, 2]))],
                'spring_prob': [float(np.min(sys_paras[:, 3])), float(np.max(sys_paras[:, 3]))]
            }
        }

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]
        self.timelength = loc.shape[2]  # 根据实际数据长度更新 timelength
        print(f"Domain {self.current_domain_id} - {data_type} data:")
        print(f"  Number of graphs: {self.num_graph}")
        print(f"  Number of atoms: {self.num_atoms}")

        # 计算域统计信息
        self.domain_stats[self.current_domain_id] = {
            'samples': self.num_graph,
            'param_range': {
                'box_size': [float(np.min(sys_paras[:, 0])), float(np.max(sys_paras[:, 0]))],
                'vel_norm': [float(np.min(sys_paras[:, 1])), float(np.max(sys_paras[:, 1]))],
                'interaction_strength': [float(np.min(sys_paras[:, 2])), float(np.max(sys_paras[:, 2]))],
                'spring_prob': [float(np.min(sys_paras[:, 3])), float(np.max(sys_paras[:, 3]))]
            }
        }

        if self.suffix == "_springs5" or self.suffix == "_charged5" or self.suffix == "_springs50" \
                or self.suffix == "_springs10" or self.suffix == "_charged10":
            # 标准化特征到 [-1, 1]（仅在启用归一化时）

            # 使用全局参数标准化
            if self.max_loc == None:
                loc, max_loc, min_loc = self.normalize_features(loc,
                                                                self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                vel, max_vel, min_vel = self.normalize_features(vel, self.num_atoms)
                print(max_loc)
                print(min_loc)
                print(max_vel)
                print(min_vel)
                self.max_loc = max_loc
                self.min_loc = min_loc
                self.max_vel = max_vel
                self.min_vel = min_vel
            else:
                loc = (loc - self.min_loc)*1.8  / (self.max_loc - self.min_loc)-0.9
                vel = (vel - self.min_vel)*1.8 / (self.max_vel - self.min_vel)-0.9



        # split data w.r.t interp and extrap, also normalize times
        if self.mode=="interp":
            loc_en,vel_en,times_en = self.interp_extrap(loc,vel,times,self.mode,data_type)
            loc_de = loc_en
            vel_de = vel_en
            times_de = times_en
        elif self.mode == "extrap":
            loc_en,vel_en,times_en,loc_de,vel_de,times_de = self.interp_extrap(loc,vel,times,self.mode,data_type)

        #Encoder dataloader
        series_list_observed, loc_observed, vel_observed, times_observed = self.split_data(loc_en, vel_en, times_en)
        if self.mode == "interp":
            time_begin = 0
        else:
            time_begin = 1
        encoder_data_loader, graph_data_loader = self.transfer_data(loc_observed, vel_observed, edges,
                                                                    times_observed, time_begin=time_begin)


        # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])
        off_diag_idx = np.array(off_diag_idx, dtype=np.int64)  # 确保是 ndarray
        off_diag_idx = torch.from_numpy(off_diag_idx).long()
        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        sys_paras = torch.tensor(sys_paras, dtype=torch.float32)
        sys_para_loader = Loader(sys_paras, batch_size=self.batch_size)

        # Decoder Dataloader
        if self.mode=="interp":
            series_list_de = series_list_observed
        elif self.mode == "extrap":
            series_list_de = self.decoder_data(loc_de,vel_de,times_de)
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]


        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)
        sys_para_loader = utils.inf_generator(sys_para_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, sys_para_loader, num_batch



    def interp_extrap(self,loc,vel,times,mode,data_type):
        if mode =="interp":
            if data_type== "test":
                # get ride of the extra nodes in testing data.
                # Create arrays with correct shape (removing last num_pre steps)
                expected_length = loc.shape[2] - self.num_pre
                loc_observed = np.zeros((loc.shape[0], loc.shape[1], expected_length, loc.shape[3]))
                vel_observed = np.zeros((vel.shape[0], vel.shape[1], expected_length, vel.shape[3]))
                times_observed = np.zeros((times.shape[0], times.shape[1], expected_length))
                
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                return loc_observed,vel_observed,times_observed/self.total_ode_step
            else:
                return loc,vel,times/self.total_ode_step


        elif mode == "extrap":# split into 2 parts and normalize t seperately

            loc_observed = [[None for _ in range(self.num_atoms)] for _ in range(self.num_graph)]
            vel_observed = [[None for _ in range(self.num_atoms)] for _ in range(self.num_graph)]
            times_observed = [[None for _ in range(self.num_atoms)] for _ in range(self.num_graph)]

            loc_extrap = [[None for _ in range(self.num_atoms)] for _ in range(self.num_graph)]
            vel_extrap = [[None for _ in range(self.num_atoms)] for _ in range(self.num_graph)]
            times_extrap = [[None for _ in range(self.num_atoms)] for _ in range(self.num_graph)]

            for i in range(self.num_graph):
                for j in range(self.num_atoms):

                    loc_observed[i][j] = loc[i][j][:self.num_condition]
                    vel_observed[i][j] = vel[i][j][:self.num_condition]
                    times_observed[i][j] = times[i][j][:self.num_condition]

                    loc_extrap[i][j] = loc[i][j][self.num_condition:self.num_condition+self.extrap_num]
                    vel_extrap[i][j] = vel[i][j][self.num_condition:self.num_condition+self.extrap_num]
                    times_extrap[i][j] = times[i][j][self.num_condition:self.num_condition+self.extrap_num]

            loc_observed = np.asarray(loc_observed)
            vel_observed = np.asarray(vel_observed)
            times_observed = np.asarray(times_observed)
            loc_extrap = np.asarray(loc_extrap)
            vel_extrap = np.asarray(vel_extrap)
            times_extrap = np.asarray(times_extrap)

            times_observed = times_observed / self.total_step_train
            times_extrap = (times_extrap - self.num_condition) / self.total_step_train

            return loc_observed,vel_observed,times_observed,loc_extrap,vel_extrap,times_extrap


    def split_data(self,loc,vel,times):
        loc_observed = np.ones_like(loc)[:, :, 1:]
        vel_observed = np.ones_like(vel)[:, :, 1:]
        times_observed = np.ones_like(times)[:, :, 1:]

        # split encoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j][1:])
                times_list.append(times[i][j][1:])

        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(loc_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series)
            # preserved_idx = sorted(
            #     np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
            preserved_idx = np.asarray(list(range(length)))
            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            vel_observed[graph_index][atom_index] = vel_list[i][preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]

            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = np.concatenate(
                (loc_series[preserved_idx], vel_list[i][preserved_idx]), axis=1)
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))

            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))


        return series_list, loc_observed, vel_observed, times_observed

    def decoder_data(self, loc, vel, times):

        # split decoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j])
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):
            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list



    def transfer_data(self, loc, vel, edges, times, time_begin=0):
        data_list = []
        graph_list = []
        edge_size_list = []

        for i in tqdm(range(self.num_graph)):
            data_per_graph, edge_data, edge_size = self.transfer_one_graph(loc[i], vel[i], edges[i], times[i],
                                                                           time_begin=time_begin)
            data_list.append(data_per_graph)
            graph_list.append(edge_data)
            edge_size_list.append(edge_size)

        print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        
        # 根据采样策略创建数据加载器
        if self.domain_sampling_strategy == 'balanced':
            data_loader = self._create_balanced_loader(data_list)
        elif self.domain_sampling_strategy == 'id_focus':
            data_loader = self._create_id_focus_loader(data_list)
        else:  # 默认使用普通的DataLoader
            data_loader = DataLoader(data_list, batch_size=self.batch_size)
            
        graph_loader = DataLoader(graph_list, batch_size=self.batch_size)
        return data_loader, graph_loader
        
    def _create_balanced_loader(self, data_list):
        """创建平衡的数据加载器，确保每个域的样本数量相等"""
        if len(self.domain_stats) <= 1:
            return DataLoader(data_list, batch_size=self.batch_size)
            
        # 找到每个域中最少的样本数
        min_samples = min(stats['samples'] for stats in self.domain_stats.values())
        
        # 对每个域进行随机采样
        balanced_data = []
        domain_indices = {}
        
        for data in data_list:
            domain_id = data.domain_id[0].item()
            if domain_id not in domain_indices:
                domain_indices[domain_id] = []
            domain_indices[domain_id].append(data)
            
        for domain_id, domain_data in domain_indices.items():
            if len(domain_data) > min_samples:
                # 随机采样
                indices = torch.randperm(len(domain_data))[:min_samples]
                balanced_data.extend([domain_data[i] for i in indices])
            else:
                balanced_data.extend(domain_data)
                
        # 打乱数据顺序
        random.shuffle(balanced_data)
        return DataLoader(balanced_data, batch_size=self.batch_size)
        
    def _create_balanced_loader(self, data_list):
        """创建平衡的数据加载器，确保每个域的样本数量相等"""
        if len(self.domain_stats) <= 1:
            return DataLoader(data_list, batch_size=self.batch_size)
            
        # 按域分组数据
        domain_data = {}
        for data in data_list:
            domain_id = data.domain_id[0].item()
            if domain_id not in domain_data:
                domain_data[domain_id] = []
            domain_data[domain_id].append(data)
        
        # 找到每个域中最少的样本数
        min_samples = min(len(samples) for samples in domain_data.values())
        
        # 对每个域进行随机采样
        balanced_data = []
        for domain_samples in domain_data.values():
            if len(domain_samples) > min_samples:
                indices = torch.randperm(len(domain_samples))[:min_samples]
                balanced_data.extend([domain_samples[i] for i in indices])
            else:
                balanced_data.extend(domain_samples)
        
        # 打乱数据顺序
        random.shuffle(balanced_data)
        return DataLoader(balanced_data, batch_size=self.batch_size)

    def _create_id_focus_loader(self, data_list):
        """创建以ID数据为主的数据加载器"""
        if len(self.domain_stats) <= 1:
            return DataLoader(data_list, batch_size=self.batch_size)
            
        # 分离ID和OOD数据
        id_data = []
        ood_data = []
        for data in data_list:
            if data.domain_id[0].item() == 0:
                id_data.append(data)
            else:
                ood_data.append(data)
                
        if not id_data or not ood_data:
            return DataLoader(data_list, batch_size=self.batch_size)
            
        # 计算每个批次中ID和OOD数据的数量
        id_batch_size = int(self.batch_size * self.id_ood_ratio)
        ood_batch_size = self.batch_size - id_batch_size
        
        # 创建混合数据集
        mixed_data = []
        id_idx = 0
        ood_idx = 0
        
        while id_idx < len(id_data) and ood_idx < len(ood_data):
            # 添加ID数据
            for _ in range(id_batch_size):
                if id_idx < len(id_data):
                    mixed_data.append(id_data[id_idx])
                    id_idx += 1
                    
            # 添加OOD数据
            for _ in range(ood_batch_size):
                if ood_idx < len(ood_data):
                    mixed_data.append(ood_data[ood_idx])
                    ood_idx += 1
                    
        # 添加剩余数据
        mixed_data.extend(id_data[id_idx:])
        mixed_data.extend(ood_data[ood_idx:])
        
        return DataLoader(mixed_data, batch_size=self.batch_size)

    def _create_domain_aware_loader(self, data_list):
        """创建域感知的数据加载器
        根据域采样策略选择合适的加载器
        """
        if self.domain_sampling_strategy == 'balanced':
            return self._create_balanced_loader(data_list)
        elif self.domain_sampling_strategy == 'id_focus':
            return self._create_id_focus_loader(data_list)
        else:
            logger.warning(f"未知的域采样策略: {self.domain_sampling_strategy}，使用默认的平衡采样")
            return self._create_balanced_loader(data_list)

    def transfer_one_graph(self,loc, vel, edge, time, time_begin=0, mask=True, forward=False):
        # Creating x : [N,D]
        # Creating edge_index
        # Creating edge_attr 
        # Creating edge_type
        # Creating y: [N], value= num_steps
        # Creating pos [N]
        # Creating domain info
        # forward: t0=0;  otherwise: t0=tN/2

        # 计算域统计信息
        try:
            if hasattr(self, 'current_domain_id'):
                domain_id = self.current_domain_id
                stats = self._compute_domain_stats(loc, vel, time, domain_id)
                logger.debug(f"成功计算域 {domain_id} 的统计信息")
        except Exception as e:
            logger.warning(f"计算域统计信息时出错: {e}")



        # compute cutting window size:
        if self.cutting_edge:
            if self.suffix == "_springs5" or self.suffix == "_charged5" or self.suffix == "_springs10" or self.suffix == "_charged10":
                # max_gap = (self.total_step - 40*self.sample_percent) /self.total_step
                max_gap = (self.total_ode_step - self.num_pre * self.sample_percent) / self.total_ode_step
            else:
                max_gap = (self.total_ode_step - 30 * self.sample_percent) / self.total_ode_step
        else:
            max_gap = 100


        if self.mode=="interp":
            forward= False
        else:
            forward=True


        y = np.zeros(self.num_atoms)
        x = list()
        x_pos = list()
        node_number = 0
        node_time = dict()
        ball_nodes = dict()

        # Creating x, y, x_pos
        for i, ball in enumerate(loc):
            loc_ball = ball
            vel_ball = vel[i]
            time_ball = time[i]

            # Creating y
            y[i] = len(time_ball)

            # Creating x and x_pos, by tranverse each ball's sequence
            for j in range(loc_ball.shape[0]):
                xj_feature = np.concatenate((loc_ball[j], vel_ball[j]))
                x.append(xj_feature)

                x_pos.append(time_ball[j] - time_begin)
                node_time[node_number] = time_ball[j]

                if i not in ball_nodes.keys():
                    ball_nodes[i] = [node_number]
                else:
                    ball_nodes[i].append(node_number)

                node_number += 1

        '''
         matrix computing
         '''
        # Adding self-loop
        edge_with_self_loop = edge + np.eye(self.num_atoms, dtype=int)

        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for i in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for i in range(len(x_pos))], axis=0)
        edge_exist_matrix = np.zeros((len(x_pos), len(x_pos)))

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if edge_with_self_loop[i][j] == 1:
                    sender_index_start = int(np.sum(y[:i]))
                    sender_index_end = int(sender_index_start + y[i])
                    receiver_index_start = int(np.sum(y[:j]))
                    receiver_index_end = int(receiver_index_start + y[j])
                    if i == j:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = 1
                    else:
                        edge_exist_matrix[sender_index_start:sender_index_end,
                        receiver_index_start:receiver_index_end] = -1

        if mask == None:
            edge_time_matrix = np.where(abs(edge_time_matrix)<=max_gap,edge_time_matrix,-2)
            edge_matrix = (edge_time_matrix + 2) * abs(edge_exist_matrix)  # padding 2 to avoid equal time been seen as not exists.
        elif forward == True:  # sender nodes are thosewhose time is larger. t0 = 0
            edge_time_matrix = np.where((edge_time_matrix >= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)
        elif forward == False:  # sender nodes are thosewhose time is smaller. t0 = tN/2
            edge_time_matrix = np.where((edge_time_matrix <= 0) & (abs(edge_time_matrix)<=max_gap), edge_time_matrix, -2) + 2
            edge_matrix = edge_time_matrix * abs(edge_exist_matrix)

        _, edge_attr_same = self.convert_sparse(edge_exist_matrix * edge_matrix)
        edge_is_same = np.where(edge_attr_same > 0, 1, 0).tolist()

        edge_index, edge_attr = self.convert_sparse(edge_matrix)
        edge_attr = edge_attr - 2
        edge_index_original, _ = self.convert_sparse(edge)



        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        edge_is_same = torch.FloatTensor(np.asarray(edge_is_same))

        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)

        # 添加域标识信息
        domain_id = torch.full((x.size(0),), self.current_domain_id, dtype=torch.long)
        is_id_domain = torch.full((x.size(0),), float(self.current_domain_id == 0), dtype=torch.float)

        graph_index_original = torch.LongTensor(edge_index_original)
        edge_data = Data(x=torch.ones(self.num_atoms), edge_index=graph_index_original)

        # 创建包含域信息的图数据
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=x_pos,
            edge_same=edge_is_same,
            domain_id=domain_id,  # 添加域ID
            is_id_domain=is_id_domain,  # 添加是否为ID域的标识
            domain_stats=self.domain_stats.get(self.current_domain_id, {})  # 添加域统计信息
        )
        edge_size = edge_index.shape[1]

        return graph_data, edge_data, edge_size

    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise. Since in human dataset, it join the data of four tags (belt, chest, ankles) into a single time series
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True)  # [including 0 ]
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])


        for b, ( tt, vals, mask) in enumerate(batch):

            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask

        # get rid of the padding timepoint
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:,1:,:]
        combined_mask = combined_mask[:,1:,:]

        combined_tt = combined_tt.float()


        # 创建批次索引，确保每个时间步都有对应的批次标识
        batch_indices = torch.arange(len(batch), dtype=torch.long)
        batch_indices = batch_indices.view(-1, 1).expand(-1, combined_vals.size(1))
        batch_indices = batch_indices.reshape(-1)
        
        # 返回字典格式
        return {
            'data': combined_vals,
            'time_steps': combined_tt,
            'mask': combined_mask,
            'batch': batch_indices
        }

    def normalize_features(self,inputs, num_balls):
        '''

        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        self.timelength = max(value_list_length)
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
        value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        # Normalize to [-1, 1]
        inputs = (inputs - min_value)*1.8  / (max_value - min_value)-0.9
        return inputs,max_value,min_value

    def convert_sparse(self,graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr

    def _compute_domain_stats(self, loc, vel, times, domain_id):
        """计算域的统计信息

        Args:
            loc: 位置数据
            vel: 速度数据
            times: 时间数据
            domain_id: 域ID

        Returns:
            dict: 包含域统计信息的字典
        """
        try:
            # 计算样本数量
            num_samples = len(loc)
            
            # 计算时间序列的统计信息
            times_mean = np.mean([np.mean(t) for t in times])
            times_std = np.std([np.std(t) for t in times])
            
            # 更新域统计信息
            self.domain_stats[domain_id] = {
                'samples': num_samples,
                'times_mean': times_mean,
                'times_std': times_std
            }
            
            # 更新域样本计数
            self.domain_sample_counts[domain_id] = num_samples
            
            return self.domain_stats[domain_id]
            
        except Exception as e:
            logger.error(f"计算域 {domain_id} 统计信息时出错: {str(e)}")
            return {}





