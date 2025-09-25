import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

	
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='a+')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

def flatten(x, dim):
	return x.reshape(x.size()[:dim] + (-1, ))


def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def get_dict_template():
	return {"data": None,
			"time_setps": None,
			"mask": None
			}
def get_next_batch_new(dataloader,device):
	data_dict = dataloader.__next__()
	#device_now = data_dict.batch.device
	return data_dict.to(device)

def get_next_batch(dataloader,device):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template()


	batch_dict["data"] = data_dict["data"].to(device)
	batch_dict["time_steps"] = data_dict["time_steps"].to(device)
	batch_dict["mask"] = data_dict["mask"].to(device)

	return batch_dict


def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	try:
		# 首先尝试使用weights_only=True（安全模式）
		checkpt = torch.load(ckpt_path, weights_only=True)
	except Exception as e:
		print(f"安全模式加载失败: {e}")
		print("尝试使用兼容模式加载模型")
		# 如果失败，使用weights_only=False（兼容模式）
		checkpt = torch.load(ckpt_path, weights_only=False)
	ckpt_args = checkpt['args']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr


def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1,
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)




def compute_loss_all_batches(model,
	encoder,graph,decoder, sys_para,
	n_batches, device,
	n_traj_samples = 1, kl_coef = 1., input_dim=4):

	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["mse"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0

	total["q_mse"] = 0
	total["v_mse"] = 0
	for i in range(input_dim):
		total["feature{}_mse".format(i)] = 0

	n_test_batches = 0

	model.eval()
	print("Computing loss... ")
	with torch.no_grad():
		for i in tqdm(range(n_batches)):
			batch_dict_encoder = get_next_batch_new(encoder, device)
			batch_dict_graph = get_next_batch_new(graph, device)
			batch_dict_decoder = get_next_batch(decoder, device)
			batch_dict_para = get_next_batch_new(sys_para, device)

			results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,batch_dict_para,
											   n_traj_samples=n_traj_samples, kl_coef=kl_coef)

			for key in total.keys():
				if key in results:
					var = results[key]
					if isinstance(var, torch.Tensor):
						var = var.detach().item()
					total[key] += var

			n_test_batches += 1

			del batch_dict_encoder,batch_dict_graph,batch_dict_decoder,batch_dict_para,results

		if n_test_batches > 0:
			for key, value in total.items():
				total[key] = total[key] / n_test_batches


	return total

#domain_ids = torch.tensor([0, 2, 0, 1], dtype=torch.long)
def get_domain_id(domain_info, data):
	domain_info_list = []
	for domain in domain_info:
		domain_info_list.append(domain.item())

	param_keys = list(domain_info_list[0].keys())
	all_matches = []
	for domain in domain_info_list:
		# 提取当前域所有参数的最小值和最大值，并构建为张量
		mins = torch.tensor([domain[key][0] for key in param_keys], device=data.device, dtype=data.dtype)
		maxs = torch.tensor([domain[key][1] for key in param_keys], device=data.device, dtype=data.dtype)

		# 使用广播机制，一次性比较整个batch的数据
		# is_in_range 的形状为 [batch_size, num_params]
		is_in_range = (data >= mins) & (data <= maxs)

		# 只有所有参数都在范围内，才算匹配该域
		# torch.all(dim=1) 会检查每一行是否都为True
		# matches_this_domain 的形状为 [batch_size]
		matches_this_domain = torch.all(is_in_range, dim=1)
		all_matches.append(matches_this_domain)

	# 3. 将所有域的匹配结果堆叠起来
	# matches_tensor 的形状为 [num_domains, batch_size]
	matches_tensor = torch.stack(all_matches)

	# 4. 找出每个样本的域ID
	# torch.argmax 在布尔张量上会返回第一个True的索引
	# 我们将 matches_tensor 转置为 [batch_size, num_domains] 以便按样本操作
	# .int() 将布尔值转为0/1
	domain_indices = torch.argmax(matches_tensor.int(), dim=0)

	# 5. 处理不匹配任何域的情况
	# 如果一个样本在所有域上都不匹配（一整列都为False），argmax会默认返回0，这是错误的。
	# 我们需要一个mask来标记那些至少匹配了一个域的样本。
	any_match = torch.any(matches_tensor, dim=0)

	# 使用 torch.where 根据mask来决定最终的ID
	# 如果any_match为True，则使用argmax找到的domain_indices；否则，设为-1。
	final_domain_ids = torch.where(any_match, domain_indices, -1)
	return final_domain_ids


