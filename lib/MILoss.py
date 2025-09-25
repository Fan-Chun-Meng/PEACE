import torch
import torch.nn.functional as F
from torch import nn


class MILoss(nn.Module):
    """
    实现了 PURE 论文中公式(11)的互信息损失。
    采用对抗学习的方式来最小化 μ 和 z 之间的互信息。
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # 定义判别器 T_gamma (一个简单的MLP)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mu_embeddings, z_embeddings):
        """
        参数:
            mu_embeddings (Tensor): 观测嵌入, (B, N, D_h)
            z_embeddings (Tensor): 提示嵌入, (B, N, D_h)
        """
        batch_size, n_particles, _ = mu_embeddings.shape

        # 1. 准备正样本对 (μ_i, z_i)
        positive_pairs = torch.cat([mu_embeddings, z_embeddings], dim=-1)

        # 2. 准备负样本对 (μ_i, z_j) where i != j
        # 通过对 z 进行 roll 操作来快速创建负样本
        z_shuffled = torch.roll(z_embeddings, shifts=1, dims=1)
        negative_pairs = torch.cat([mu_embeddings, z_shuffled], dim=-1)

        # 将正负样本拼接在一起
        all_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)

        # 3. 通过判别器得到分数
        scores = self.discriminator(all_pairs)

        # 4. 根据公式(11)计算损失
        # sp(x) = log(1 + e^x)
        # 正样本分数: scores_pos, 负样本分数: scores_neg
        scores_pos, scores_neg = torch.split(scores, batch_size, dim=0)

        # 论文中的目标是 max_gamma E[sp(-T)] + E[-sp(T)]
        # 在训练主模型时，我们是 min E[T] (愚弄判别器)，所以我们希望正样本分数变低
        # 这里我们直接返回一个可以被最小化的损失项
        # 我们希望 T(pos) -> inf, T(neg) -> -inf for discriminator
        # We want to fool discriminator, so T(pos) -> -inf for generator
        loss = -torch.mean(F.logsigmoid(scores_pos) + F.logsigmoid(-scores_neg))

        return loss