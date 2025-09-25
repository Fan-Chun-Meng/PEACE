import torch
import torch.nn as nn
import math


# 假设 ScaledDotProductAttention 已定义
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v):
        dim_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dim_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class PromptInitializer(nn.Module):
    """
    多视角上下文探索模块 (Prompt Initializer) - 最终版

    该模块根据系统的初始状态（位置、速度）和物理参数为每个粒子生成一个初始提示嵌入(z^0)。
    它专为粒子/图结构系统（如弹簧质点）进行了适配。

    """

    def __init__(self,
                 initial_pos_dim: int = 2,
                 particle_state_dim: int = 4,
                 param_dim: int = 3,  # 新增: 物理参数的维度
                 hidden_dim: int = 128,
                 num_attention_layers: int = 2,
                 num_heads: int = 4):
        """
        初始化 PromptInitializer 模块的各个层。

        参数:
            initial_pos_dim (int): 粒子初始位置的维度。
            particle_state_dim (int): 描述粒子状态的向量维度。
            param_dim (int): 物理参数向量的维度。
            hidden_dim (int): 模型中大多数层的隐藏维度。
            num_attention_layers (int): 自注意力堆栈中的层数 (论文中的 L)。
            num_heads (int): 多头注意力机制中的头数。
        """
        super().__init__()

        # [cite_start]1. 输入嵌入层: 将位置和状态信息映射到隐藏空间 [cite: 126]
        self.pos_embedder = nn.Linear(initial_pos_dim, hidden_dim)
        self.state_embedder = nn.Linear(particle_state_dim, hidden_dim)

        # 新增: 参数嵌入器，用于编码物理参数
        self.parameter_embedder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # [cite_start]2. 自注意力堆栈: 用于在所有粒子间传播信息，学习上下文表示 [cite: 126, 128]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.self_attention_stack = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers
        )

        # [cite_start]3. 提示检索注意力模块: 从全局上下文中为每个粒子检索专属的提示 [cite: 142]
        self.prompt_retriever_q = nn.Linear(hidden_dim, hidden_dim)
        self.prompt_retriever_k = nn.Linear(hidden_dim, hidden_dim)
        self.prompt_retriever_v = nn.Linear(hidden_dim, hidden_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, initial_state: torch.Tensor, initial_pos: torch.Tensor, phy_params: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播，生成初始提示嵌入 z^0。

        参数:
            initial_state (torch.Tensor): 粒子的初始状态。 (B, N, D_state)
            initial_pos (torch.Tensor): 粒子的初始位置。 (B, N, D_pos)
            phy_params (torch.Tensor): 系统的物理参数。 (B, D_param)

        返回:
            torch.Tensor: 每个粒子的初始提示嵌入 z^0。 (B, N, D_hidden)
        """

        # [cite_start]Step 1.1: 生成初始聚合嵌入 [cite: 126-128]
        pos_embedding = self.pos_embedder(initial_pos)
        state_embedding = self.state_embedder(initial_state)
        aggregated_embedding = pos_embedding * state_embedding

        # [cite_start]Step 1.2: 自注意力信息增强 [cite: 128]
        context_embedding = self.self_attention_stack(aggregated_embedding)

        # --- 核心修改：注入参数信息 ---
        # 1. 将物理参数编码
        param_embedding = self.parameter_embedder(phy_params)  # (B, D_h)

        # [cite_start]2. 将参数嵌入添加到每个粒子的上下文中 [cite: 136-137]
        #    使用 unsqueeze(1) 将其从 (B, D_h) 扩展为 (B, 1, D_h) 以便广播
        context_with_params = context_embedding + param_embedding.unsqueeze(1)

        # [cite_start]Step 1.3: 检索初始提示 [cite: 142]
        # Query: 仍然使用与位置相关的信息
        # Key/Value: 使用融合了物理参数的、更丰富的上下文信息
        q = self.prompt_retriever_q(pos_embedding)
        k = self.prompt_retriever_k(context_with_params)
        v = self.prompt_retriever_v(context_with_params)

        initial_prompt = self.attention(q, k, v)

        return initial_prompt