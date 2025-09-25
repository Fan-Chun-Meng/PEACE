import torch
import torch.nn as nn


class FNO1dLayer(nn.Module):
    """
    一维傅里叶神经算子层。
    它包含一个谱卷积（在频域中的线性变换）和一个空间域的线性变换。
    """

    def __init__(self, in_channels, out_channels, num_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes  # 需要保留的傅里叶模态数

        # 频域中的线性变换权重
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, num_modes, dtype=torch.cfloat))

        # 空间域的线性变换（用于跳过连接）
        self.spatial_linear = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x 形状: (batch_size, num_particles, in_channels)
        # -> (batch_size, in_channels, num_particles)
        x = x.permute(0, 2, 1)

        # 空间域的跳过连接
        skip_connection = self.spatial_linear(x)

        # 1. 傅里叶变换
        x_fft = torch.fft.rfft(x, n=x.size(-1))

        # 2. 谱卷积（只对低频模态操作）
        out_fft = torch.zeros(x.size(0), self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        # (B, I, M) * (I, O, M) -> (B, O, M)
        out_fft[:, :, :self.num_modes] = torch.einsum("bim,iom->bom", x_fft[:, :, :self.num_modes], self.weights)

        # 3. 逆傅里叶变换
        x_ifft = torch.fft.irfft(out_fft, n=x.size(-1))

        # 加上跳过连接并应用激活函数
        output = torch.nn.functional.gelu(x_ifft + skip_connection)

        # -> (batch_size, num_particles, out_channels)
        return output.permute(0, 2, 1)

class ViT_Layer(nn.Module):
    """
    Vision Transformer 层。对于粒子数据，这等价于一个标准的 Transformer Encoder。
    """
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x 形状: (batch_size, num_particles, hidden_dim)
        return self.transformer(x)

class MLPHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.head(x)

class ResidualHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out + residual)  # 残差连接
        return out

class BackboneDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, fno_modes, vit_heads, vit_layers):
        super().__init__()

        # 新增: 在分支前添加 LayerNorm 以稳定数值
        self.norm_fno = nn.LayerNorm(hidden_dim)
        self.norm_vit = nn.LayerNorm(hidden_dim)
        self.ResidualHead = ResidualHead(hidden_dim, output_dim)
        self.MLPHead = MLPHead(hidden_dim, output_dim)
        self.fno_branch = FNO1dLayer(hidden_dim, hidden_dim, fno_modes)
        self.vit_branch = ViT_Layer(hidden_dim, vit_heads, vit_layers)

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.prediction_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 1. 先进行归一化
        x_fno = self.norm_fno(x)
        x_vit = self.norm_vit(x)

        # 2. 通过两个并行分支
        fno_out = self.fno_branch(x_fno)
        vit_out = self.vit_branch(x_vit)

        # 3. 拼接、融合、预测
        concatenated_features = torch.cat([fno_out, vit_out], dim=-1)
        fused_features = self.fusion_layer(concatenated_features)
        prediction = self.MLPHead(fused_features)

        return prediction