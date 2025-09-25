
import torch
from torch import nn
from torchdiffeq import odeint

from lib.BackboneDecoder import BackboneDecoder
from lib.BackboneEncoder import BackboneEncoder
from lib.ODEFunc import ODEFunc, LinearInterpolator
from lib.PromptInitializer import PromptInitializer

class DS_ODE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prediction_len = config['prediction_len']
        self.num_balls = config['num_particles']
        self.history_len = config['history_len']
        self.num_prompts = config['num_prompts']

        # Prompt ensemble
        self.prompt_ensemble = nn.ModuleList()
        for _ in range(self.num_prompts):
            branch = nn.ModuleDict({
                'prompt_initializer': PromptInitializer(
                    config['pos_dim'],
                    config['state_dim'],
                    config['param_dim'],
                    config['hidden_dim'],
                    2,
                    4
                )
            })
            self.prompt_ensemble.append(branch)

        # ODE function + interpolator
        self.ODEFunc = ODEFunc(config['hidden_dim'], config['state_dim'])
        self.interpolator = LinearInterpolator()

        # Backbone encoder/decoder
        self.backbone_encoder = BackboneEncoder(config['state_dim'], config['hidden_dim'])
        self.fusion_projection = nn.Linear(config['hidden_dim'] * 2, config['hidden_dim'])

        # Backbone decoder 参数从 config 读取（给默认值保证兼容）
        dec_layers = config.get('decoder_layers', 6)
        dec_heads = config.get('decoder_heads', 4)
        dec_ff = config.get('decoder_ffn', 2)
        self.backbone_decoder = BackboneDecoder(
            config['hidden_dim']*self.history_len,
            config['state_dim']*self.prediction_len,
            dec_layers,
            dec_heads,
            dec_ff
        )

        # === Per-prompt gates (可选) ===
        # 每个 prompt 一个 gate: 输入 hidden_dim，输出 hidden_dim（通过 sigmoid 缩放）
        self.prompt_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['hidden_dim'], config['hidden_dim']),
                nn.Tanh(),
                nn.Linear(config['hidden_dim'], config['hidden_dim']),
                nn.Sigmoid()
            )
            for _ in range(self.num_prompts)
        ])

    # 在 forward 中接收 phy_params
    def forward(self, batch, mode='train'):
        """
        Forward for multi-prompt DS ODE.
        - 支持 K prompts (self.num_prompts)
        - 若在 __init__ 中添加 prompt_gates (ModuleList)，则自动使用 per-prompt gating
        - teacher forcing: 若 batch 中含 'decoder' 且 config 中有 'tf_rate'，会按概率混合 ground-truth 与 model prediction
        - collect_steps: config.get('collect_steps', 1)  -> 返回前 collect_steps 个 (mu,z) 以用于 MI 等正则
        """
        device = next(self.parameters()).device

        # 基本数据与形状
        encoder = batch.get('encoder')
        data = encoder.x.to(device)
        phy_params = batch.get('sys_para')
        batch_size = encoder.batch_size
        K = self.num_prompts
        history_len = self.history_len
        num_balls = self.num_balls

        # 形状: B x N x T x D
        history_observations = data.view(batch_size, num_balls, history_len, -1).to(device)

        # 保持每个 prompt 的独立历史（clone 避免引用同一 tensor）
        history_observations_list = [history_observations.clone() for _ in range(K)]

        # times on device
        history_times = torch.arange(history_len, dtype=torch.float32, device=device)

        predictions_list = []
        final_last_mu = []
        final_z_avg = []

        # 用于收集前若干步 mu/z（默认 collect_steps=1 即 t==0）
        collect_steps = self.config.get('collect_steps', 1)
        collect_count = 0

        # teacher forcing rate: 可以在 config 指定初始值，并在训练中动态调整 (train_step 负责更新 tf_rate)
        tf_rate = float(self.config.get('tf_rate', 0.0)) if mode == 'train' else 0.0
        # 如果 batch 提供了 decoder ground truth（形状需能匹配 prediction），将尝试使用
        has_gt = 'decoder' in batch and batch['decoder'] is not None

        # 主循环：自回归预测
        # for t in range(self.prediction_len):
        # 1) 每个 prompt 用自己的 initializer 产生 z0
        z0_i_list = []
        for i in range(K):
            # initial_state: shape (B, N, state_dim)
            initial_state = history_observations_list[i][:, :, 0, :].to(device)
            initial_pos = initial_state[:, :, :self.config['pos_dim']]
            prompt_branch = self.prompt_ensemble[i]
            z0_i = prompt_branch['prompt_initializer'](initial_state, initial_pos, phy_params)
            z0_i_list.append(z0_i)

        # 2) 把所有 prompt 的 z0 串到 batch 维度上并演化
        # 假设 z0_i 的形状 (B, N, z_dim). 我们 cat 成 (K*B, N, z_dim)
        z0_cat = torch.cat(z0_i_list, dim=0)
        # 确保 ODEFunc 拥有必要的上下文（device/observations/times）
        self.ODEFunc.interpolator = self.interpolator
        # 注意：ODEFunc 里若需要 observations/obs_times，应接收按 (K*B) 对齐的 history_observations
        # 这里我们将 history_observations_list 拼成 (K*B, N, T, D) 的形状并放到 self.ODEFunc
        history_cat = torch.cat(history_observations_list, dim=0).to(device)
        self.ODEFunc.observations = history_cat
        self.ODEFunc.obs_times = history_times

        # odeint -> 输出形状 (T, K*B, N, z_dim)；permute -> (K*B, N, T, z_dim) 再取最后 timestep
        z_t = odeint(self.ODEFunc, z0_cat, history_times, method='rk4', options=dict(step_size=0.1))
        z_t = z_t.permute(1, 2, 0, 3)  # (K*B, N, T, z_dim)
        last_z = z_t[:, :, -1, :]  # (K*B, N, z_dim)

        # 3) 共享 backbone encoder 得到 mu embedding（输入 history_cat）
        mu_embeddings = self.backbone_encoder(history_cat)  # (K*B, N, T, hidden)
        last_mu = mu_embeddings[:, :, -1, :]  # (K*B, N, hidden)

        # 4) 融合并投影
        fused_embedding = torch.cat([z_t, mu_embeddings], dim=-1)  # (K*B, N, hidden*2)
        projected_embedding = self.fusion_projection(fused_embedding)  # (K*B, N, hidden)

        # 4.1 可选 per-prompt gating/adaptation（如果 __init__ 中定义了 prompt_gates）
        # 需要把 projected_embedding 按 batch_size 分段，交给对应 gate
        if hasattr(self, 'prompt_gates') and isinstance(self.prompt_gates, nn.ModuleList) and len(
                self.prompt_gates) == K:
            # split 成 K 段，每段 shape (B, N, hidden)
            proj_splits = torch.split(projected_embedding, batch_size, dim=0)
            gated_chunks = []
            for i in range(K):
                # gate expects (B, N, hidden) --> 先 flatten时间/particles维或直接 apply pointwise MLP
                # 这里对每 time/particle 单独应用 gate：reshape -> (B*N, hidden)
                chunk = proj_splits[i]
                B, N, T, H = chunk.shape
                chunk_flat = chunk.reshape(B * N * T, H)
                gate_out = self.prompt_gates[i](chunk_flat)  # (B*N, H) or (B*N, 1)
                gated = (chunk_flat * gate_out).reshape(B, N, T, H)
                gated_chunks.append(gated)
            projected_embedding = torch.cat(gated_chunks, dim=0)  # back to (K*B, N, hidden)

        # 5) decoder -> 得到 prediction (K*B, N, pred_dim)
        prediction = self.backbone_decoder(projected_embedding.view(K*batch_size,num_balls,-1))  # (K*B, N, out_dim)
        prediction = prediction.view(K*batch_size,num_balls,-1,4)
        # 6) 将 prediction 切回每个 prompt 对应的 batch 分段，收集到 predictions_list
        preds_per_prompt = torch.split(prediction, batch_size, dim=0)  # list of K tensors (B, N, out_dim)
        predictions_list.extend(preds_per_prompt)

        # 7) 构造 new_prediction_step（用于下一个时间步历史），并作 teacher forcing / detach 处理
        # 尝试从 batch['decoder'] 获取 gt（若存在）
        if has_gt:
            # 假设 batch['decoder'] 格式可索引到 t 步的 ground truth 并能 reshape 为 (B, N, out_dim)
            try:
                gt_t = batch['decoder']['data'][:, t, ...].to(device)
                gt_t = gt_t.view(batch_size, num_balls, -1, 4)# shape must match prediction per time-step
            except Exception:
                gt_t = None
        else:
            gt_t = None

        # new_prediction_step shape: (K*B, N, out_dim) to be concatenated into history (which expects (K*B, N, 1, out_dim))
        new_prediction_step = prediction  # (K*B, N, out_dim)

        if mode == 'train' and tf_rate > 0.0 and gt_t is not None:
            # 对每个样本以概率 tf_rate 用 GT，否则用 model prediction
            # 这里做逐-sample 混合：构造 mask shape (B,)
            rand_vals = torch.rand(batch_size, device=device)
            use_gt_mask = (rand_vals < tf_rate).float()  # 1 -> use gt for that sample
            # expand to (B, N, out_dim)
            use_gt_mask = use_gt_mask.view(batch_size, 1, 1).expand(-1, num_balls, gt_t.shape[-1])
            # apply per-prompt: preds_per_prompt are (B,N,out), so we combine each chunk with gt according to mask
            mixed_chunks = []
            for i in range(K):
                pred_chunk = preds_per_prompt[i]  # (B,N,out)
                if gt_t is not None and gt_t.shape == pred_chunk.shape:
                    mixed = pred_chunk * (1.0 - use_gt_mask) + gt_t * use_gt_mask
                else:
                    mixed = pred_chunk  # fallback if shapes mismatch
                mixed_chunks.append(mixed)
            # cat back to (K*B, N, out_dim)
            new_prediction_step = torch.cat(mixed_chunks, dim=0)
        else:
            # 默认用 model 预测，但为稳定训练可 detach（减少误差回传到 decoder 内部共享模块）
            new_prediction_step = new_prediction_step.detach()

        # 8) 更新 history_cat (K*B, N, T, D) 与 history_observations_list 每个 prompt 的历史
        #new_pred_unsq = new_prediction_step.unsqueeze(2)  # (K*B, N, 1, out_dim)

        # history_cat 也是按 K*B 维度组织：先把最老的 timestep 弹出，再 cat 新步
        #history_cat = history_cat[:, :, 1:, :].contiguous()
        #history_cat = torch.cat([history_cat, new_pred_unsq], dim=2)  # (K*B, N, T, D)

        # 更新每个 prompt 的视图（split 回去）
        #spl = torch.split(history_cat, batch_size, dim=0)  # list len K
        # for i in range(K):
        #     history_observations_list[i] = spl[i]

        # 9) 收集前 collect_steps 个 mu/z 用于后续正则（例如 MI loss）
        if collect_count < collect_steps:
            # last_mu and last_z 在 (K*B, N, hidden) 形式，按 batch split
            mu_splits = torch.split(last_mu, batch_size, dim=0)
            z_splits = torch.split(last_z, batch_size, dim=0)
            final_last_mu.extend(list(mu_splits))
            final_z_avg.extend(list(z_splits))
            collect_count += 1

        # 返回与之前兼容： predictions_list 长度 = K * T，每项 shape (B, N, out_dim)
        return predictions_list, final_last_mu, final_z_avg


