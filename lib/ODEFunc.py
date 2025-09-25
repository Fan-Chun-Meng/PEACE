import torch
import torch.nn as nn


# (LinearInterpolator 类保持不变，这里省略)
class LinearInterpolator(nn.Module):
    """
    线性插值器（修正版）。

    这个版本更稳健，可以正确处理来自 torchdiffeq 求解器的
    向量化时间输入 (t_query 是一个张量)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, times: torch.Tensor, observations: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        """
        参数:
            times (Tensor): 离散的时间点, 1D 张量, e.g., [0, 1, ..., 12]
            observations (Tensor): 对应时间的观测值, (B, N, T, D_obs)
            t_query (Tensor): 需要查询的时间点, 可以是标量或1D张量。
        """
        device = observations.device
        times = times.to(device)

        # 查找 t_query 左右两侧的索引
        right_indices = torch.searchsorted(times, t_query).clamp(max=len(times) - 1)
        left_indices = (right_indices - 1).clamp(min=0)

        # 处理 t_query 正好落在时间点上的情况，确保左右索引不同
        # 除非 t_query 是第一个时间点
        is_on_grid = (times[left_indices] == t_query) & (left_indices > 0)
        left_indices[is_on_grid] -= 1

        # 获取左右端点的时间和观测值
        t_left, t_right = times[left_indices], times[right_indices]
        # 使用高级索引，即使 left_indices 是张量也能正确工作
        obs_left = observations[:, :, left_indices, :]
        obs_right = observations[:, :, right_indices, :]

        # 计算插值权重
        time_diff = (t_right - t_left)
        # 避免除以零
        time_diff[time_diff == 0] = 1e-6

        weight = (t_query - t_left) / time_diff

        # --- 核心修正 ---
        # 将 weight 的形状调整为可以正确广播到 (obs_right - obs_left) 的形状
        # obs_left/right 的形状是 (B, N, K, D_obs)，其中 K 是 t_query 的长度
        # weight 的形状是 (K,)，需要变为 (1, 1, K, 1)
        if t_query.dim() > 0:  # 如果 t_query 是一个向量
            weight = weight.view(1, 1, -1, 1)

        # 线性插值
        interpolated_obs = obs_left + weight * (obs_right - obs_left)

        # 如果 t_query 是标量，输出需要挤压掉多余的维度
        if t_query.dim() == 0:
            interpolated_obs = interpolated_obs.squeeze(2)

        return interpolated_obs


class ODEFunc(nn.Module):


    def __init__(self, hidden_dim: int, obs_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        self.relation_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(obs_dim, hidden_dim)

        self.aggregation_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.interpolator = None
        self.obs_times = None
        self.observations = None

    def forward(self, t, z):
        batch_size, n_particles, _ = z.shape

        if self.interpolator is None or self.observations is None:
            raise ValueError("Interpolator and observations must be set before calling forward.")
        s_t = self.interpolator(self.obs_times, self.observations, t)

        z_i = z.unsqueeze(2).repeat(1, 1, n_particles, 1)
        z_j = z.unsqueeze(1).repeat(1, n_particles, 1, 1)
        s_j = s_t.unsqueeze(1).repeat(1, n_particles, 1, 1)

        query = self.query_proj(z_i)
        key = self.key_proj(s_j)

        attn_scores = (query * key).sum(dim=-1) / (self.hidden_dim ** 0.5)

        # ############# 唯一的修改在这里 #############
        # 修正前 (错误)
        # identity_mask = torch.eye(n_particles, device=z.device).unsqueeze(0).unsqueeze(0)
        # 修正后 (正确)
        identity_mask = torch.eye(n_particles, device=z.device)
        # ###########################################

        attn_scores.masked_fill_(identity_mask.bool(), -float('inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        z_pairs = torch.cat([z_i, z_j], dim=-1)
        relation_term = self.relation_mlp(z_pairs)

        weighted_relations = attn_weights.unsqueeze(-1) * relation_term
        aggregated_info = weighted_relations.sum(dim=2)

        dz_dt = self.aggregation_mlp(aggregated_info)

        return dz_dt