import torch
import torch.nn as nn


class BackboneEncoder(nn.Module):
    """
    基础预测模型的编码器部分。

    该模块负责将输入的低维观测序列 (s_i^t) 编码为高维的
    观测嵌入 (μ_i^t)，为后续的解码和预测做准备。

    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化 BackboneEncoder 的层。

        参数:
            input_dim (int): 输入观测向量的维度 (例如 4 for [pos_x, pos_y, vel_x, vel_y])。
            hidden_dim (int): 输出的观测嵌入的维度，即模型的核心隐藏维度。
        """
        super().__init__()

        # 定义一个简单的两层 MLP 作为编码器 (phi^enc)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # 使用 GELU 激活函数，在 Transformer 类模型中常用
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, history_observations: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，生成观测嵌入。

        参数:
            history_observations (torch.Tensor): 包含历史轨迹的张量。
                                                 形状: (batch_size, num_particles, num_timesteps, input_dim)

        返回:
            torch.Tensor: 编码后的观测嵌入 μ^t。
                          形状: (batch_size, num_particles, num_timesteps, hidden_dim)
        """
        # MLP 会自动地作用在最后一个维度 (input_dim) 上，
        # 将其从 input_dim 映射到 hidden_dim。
        # 其他维度 (batch, num_particles, num_timesteps) 保持不变。
        observation_embeddings = self.encoder_mlp(history_observations)

        return observation_embeddings