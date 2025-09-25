#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TrajectoryLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super(TrajectoryLoss, self).__init__()
        self.loss_type = loss_type
        
    def forward(self, predictions, targets, mask=None, norm_params=None):
        """
        Args:
            predictions: [batch_size, num_balls, time_length, 4]
            targets: [batch_size, num_balls, time_length, 4]
            mask: [batch_size, num_balls, time_length, 4] 可选的掩码
            norm_params: 归一化参数，包含max_val和min_val (为了兼容性保留，但不使用)
        Returns:
            total_loss: 总损失
            position_loss: 位置损失 (q)
            velocity_loss: 速度损失 (v)
        """
        # 注意：为了与baseline保持一致，我们在归一化数据上直接计算MSE
        # 不进行反归一化处理，这样MSE结果才能与baseline可比较
            
        # 创建默认掩码
        if mask is None:
            mask = torch.ones_like(predictions)
            

        # 将batch_size和num_balls合并为n_traj
        batch_size, num_balls = predictions.shape[:2]
        predictions = predictions.view(1, batch_size * num_balls, -1, 4).repeat(3, 1, 1, 1)
        targets = targets.view(1, batch_size * num_balls, -1, 4).repeat(3, 1, 1, 1)
        mask = mask.view(1, batch_size * num_balls, -1, 4).repeat(3, 1, 1, 1)
        
        # 计算位置MSE
        pred_pos = predictions[..., :2]  # [1, n_traj, time_length, 2]
        target_pos = targets[..., :2]
        pos_mask = mask[..., :2]
        

        pos_squared_error = ((pred_pos - target_pos) ** 2) * pos_mask  # mse函数
        # 在时间维度上求和 [1, n_traj, 2]
        pos_sum = torch.sum(pos_squared_error, dim=2)
        # 计算每个特征的时间长度 [1, n_traj, 2]
        time_length_pos = torch.sum(pos_mask, dim=2)
        # 避免除零
        time_length_pos = torch.clamp(time_length_pos, min=1.0)
        # 按时间长度归一化 [1, n_traj, 2]
        pos_normalized = torch.div(pos_sum, time_length_pos)
        # 先对特征维度取平均 [1, n_traj]
        pos_normalized = torch.mean(pos_normalized, dim=-1)
        # 分 x/y 方向 MSE
        pos_mse_x = pos_normalized[..., 0].mean()  # 平均所有 traj
        pos_mse_y = pos_normalized[..., 1].mean()
        # 再对n_traj维度取平均得到最终损失
        position_loss = torch.mean(pos_normalized)
        
        # 计算速度MSE
        pred_vel = predictions[..., 2:]  # [1, n_traj, time_length, 2]
        target_vel = targets[..., 2:]
        vel_mask = mask[..., 2:]
        

        vel_squared_error = ((pred_vel - target_vel) ** 2) * vel_mask  # mse函数
        # 在时间维度上求和 [1, n_traj, 2]
        vel_sum = torch.sum(vel_squared_error, dim=2)
        # 计算每个特征的时间长度 [1, n_traj, 2]
        time_length_vel = torch.sum(vel_mask, dim=2)
        # 避免除零
        time_length_vel = torch.clamp(time_length_vel, min=1.0)
        # 按时间长度归一化 [1, n_traj, 2]
        vel_normalized = torch.div(vel_sum, time_length_vel)
        # 先对特征维度取平均 [1, n_traj]
        vel_normalized = torch.mean(vel_normalized, dim=-1)
        # 分 x/y 方向 MSE
        vel_mse_x = vel_normalized[..., 0].mean()
        vel_mse_y = vel_normalized[..., 1].mean()
        # 再对n_traj维度取平均得到最终损失
        velocity_loss = torch.mean(vel_normalized)
        
        # 总损失为位置和速度损失的平均
        total_loss = (position_loss + velocity_loss) / 2

        diff = (predictions - targets) ** 2
        masked_diff = diff * mask
        # 每个 feature 按时间维度求和，再除以有效时间步数
        time_len = mask.sum(dim=1).clamp(min=1.0)  # [B, N, 4]
        mse_per_feature = masked_diff.sum(dim=1) / time_len  # [B, N, 4]
        # 平均所有特征、所有粒子、所有 batch
        total_loss = mse_per_feature.mean()
        return total_loss, position_loss, velocity_loss, pos_mse_x, pos_mse_y, vel_mse_x, vel_mse_y


def create_fine_grained_target(batch, num_balls, time_length):
    """
    从批次数据创建细粒度目标
    
    Args:
        batch: PyG批次数据
        num_balls: 每个图的ball数量
        time_length: 时间序列长度
        
    Returns:
        targets: [batch_size, num_balls, time_length, 4]
    """
    batch_size = batch['data'].shape[0]//num_balls
    
    # 从target_loc和target_vel构建目标
    if hasattr(batch, 'target_loc') and hasattr(batch, 'target_vel'):
        # 获取每个图的目标位置和速度
        targets = torch.zeros(batch_size, num_balls, time_length, 4, device=batch.x.device)

        # 填充每个图的目标数据
        for i in range(batch_size):
            graph_mask = (batch.batch == i)
            # 获取当前图的位置和速度目标
            loc = batch.target_loc[graph_mask]
            vel = batch.target_vel[graph_mask]

            # 根据数据加载器的实现，target_loc和target_vel的格式有两种可能：
            # 1. 来自multi_domain_dataloader: [num_balls*time_length, 2] (2D张量)
            # 2. 来自new_dataLoader: [num_balls, time_length, 2] (3D张量)

            if loc.dim() == 2:  # 如果是2维张量 [num_balls*time_length, 2]
                # 重塑为3维张量 [num_balls, time_length, 2]
                try:
                    loc = loc.view(num_balls, time_length, 2)
                    vel = vel.view(num_balls, time_length, 2)
                except RuntimeError as e:
                    # 如果重塑失败，说明数据大小不匹配，尝试其他处理方式
                    print(f"Warning: Failed to reshape target data for graph {i}: {e}")
                    print(f"loc shape: {loc.shape}, expected: [{num_balls*time_length}, 2]")
                    # 如果数据大小不匹配，跳过当前图
                    continue
            elif loc.dim() == 3:  # 如果已经是3维张量 [num_balls, time_length, 2]
                pass  # 保持原样
            else:
                print(f"Warning: Unexpected target data dimension {loc.dim()} for graph {i}")
                continue

            # 确保数据维度正确
            if loc.shape[0] >= num_balls and loc.shape[1] >= time_length:
                # 合并位置和速度
                targets[i, :num_balls, :time_length, :2] = loc[:num_balls, :time_length]
                targets[i, :num_balls, :time_length, 2:] = vel[:num_balls, :time_length]
            else:
                print(f"Warning: Target data size mismatch for graph {i}: loc shape {loc.shape}, expected at least [{num_balls}, {time_length}, 2]")

        return targets, None
    else:
        # 如果没有目标数据，从现有的位置数据构建目标
        # new_dataLoader.py中的数据格式：x包含位置和速度信息，pos包含位置信息
        #print("Info: No target_loc/target_vel found, constructing targets from existing data")

        targets = batch['data'].view(batch_size,num_balls,time_length,4)
        
        # 尝试从pos和x数据构建目标
        # if hasattr(batch, 'pos') and batch.pos is not None:
        #     for i in range(batch_size):
        #         graph_mask = (batch.batch == i)
        #         pos_data = batch.pos[graph_mask]  # 位置数据
        #         x_data = batch.x[graph_mask]  # 节点特征数据，包含位置和速度
        #         # 确保使用正确的时间长度
        #         x_reshaped = x_data.view(num_balls, -1, 4)
        #         actual_time_steps = x_reshaped.shape[1]
        #         if actual_time_steps >= time_length:
        #             targets[i] = x_reshaped[:, -time_length:, :]
        #         else:
        #             # 如果实际时间步数不足，用零填充
        #             targets[i, :, :actual_time_steps, :] = x_reshaped
        #             targets[i, :, actual_time_steps:, :] = 0
                # 安全的维度检查

        
        return targets, None