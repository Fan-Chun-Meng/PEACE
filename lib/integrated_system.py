"""Integrated PromptODE System

This module contains the IntegratedPromptODESystem class that integrates
PromptODE training and inference system with dual-level optimization,
gradient agreement selection, ensemble inference, and meta-learning scheduling.
"""

from lib.MILoss import MILoss
import torch

import logging
import torch.nn.functional as F # 用于计算余弦相似度
from .enhanced_promptode import create_fine_grained_target, TrajectoryLoss


class IntegratedPromptODESystem:

    def __init__(self, model, device, args, prompt_lr=5e-4, backbone_lr=1e-4):

        self.model = model
        self.device = device
        self.args = args

        # --- 参数分组: Prompt vs Backbone ---
        prompt_params, backbone_params = [], []
        for name, param in self.model.named_parameters():
            if 'prompt_ensemble' in name or 'prompt_gates' in name:
                prompt_params.append(param)
            else:
                backbone_params.append(param)

        # --- 双优化器 ---
        self.optimizer_backbone = torch.optim.AdamW(backbone_params, lr=backbone_lr)
        self.optimizer_prompts = torch.optim.AdamW(prompt_params, lr=prompt_lr)

        # --- 初始化 EMA 分数用于 prompt 权重 ---
        num_prompts = getattr(self.model, 'num_prompts', len(getattr(self.model, 'prompt_ensemble', [])))
        self.ema_beta = 0.9
        # register_buffer 确保 EMA scores 在 model.to(device) 时也会移动
        self.model.register_buffer('prompt_ema_scores', torch.ones(num_prompts, device=self.device))

        # 推理阶段 softmax 温度
        self.inference_temperature = 0.5

        logging.info(
            f"IntegratedPromptODESystem initialized with {num_prompts} prompts, "
            f"dual optimizers, EMA tracking, device={self.device}"
        )

    def train_step(self, batch, data_loader, batch_step, epoch):
        """Execute training step with gradient consistency"""

        self.model.train()
        self.optimizer_backbone.zero_grad()
        self.optimizer_prompts.zero_grad()

        # === 构造 targets ===
        targets, _ = create_fine_grained_target(
            batch['decoder'],
            num_balls=getattr(data_loader, 'n_balls', self.model.num_balls),
            time_length=self.model.prediction_len
        )

        criterion = TrajectoryLoss(loss_type='mse')
        mi_loss_fn = MILoss(64).to(self.device)

        # === Forward ===
        model_output = self.model(batch, mode='train')
        predictions_group, first_step_mu_list, first_step_z_list = model_output

        # === 聚合每个 prompt 的输出 ===
        # predictions_group = []
        # for i in range(self.model.num_prompts):
        #     predictions = predictions_list[i::self.model.num_prompts]
        #     predictions_group.append(torch.stack(predictions, dim=0).permute(1, 2, 0, 3))

        predictions_stack = torch.stack(predictions_group, dim=0)
        first_step_mu_stack = torch.stack(first_step_mu_list, dim=0)
        first_step_z_stack = torch.stack(first_step_z_list, dim=0)

        # === 权重 softmax ===
        weights = F.softmax(self.model.prompt_ema_scores / self.inference_temperature, dim=0).to(self.device)

        # === 加权平均 ===
        predictions_avg = torch.sum(predictions_stack * weights.view(-1, 1, 1, 1, 1), dim=0)
        first_step_mu_avg = torch.sum(first_step_mu_stack * weights.view(-1, 1, 1, 1), dim=0)
        first_step_z_avg = torch.sum(first_step_z_stack * weights.view(-1, 1, 1, 1), dim=0)

        batch_data = batch['encoder'].x.view(batch['encoder'].batch_size, 10, -1, 4).to(self.device)
        # === Backbone 损失 ===pre, tar, hist=None, threshold=0.01, save_list=None, mode="train"
        #filter_and_save_samples(pre = predictions_avg, tar = targets, hist = batch_data, mode='train')
        total_loss, pos_loss, vel_loss, pos_mse_x, pos_mse_y, vel_mse_x, vel_mse_y = criterion(predictions_avg, targets)
        mi_loss = mi_loss_fn(first_step_mu_avg, first_step_z_avg)
        final_loss_for_backbone = total_loss
        final_loss_for_backbone.backward(retain_graph=True)

        # === Prompt-specific 梯度 ===
        prompt_losses = [criterion(pred, targets)[0] for pred in predictions_group]
        prompt_grads_k = []
        for i, loss in enumerate(prompt_losses):
            grad = torch.autograd.grad(loss,
                                       self.model.prompt_ensemble[i].parameters(),
                                       retain_graph=True)
            prompt_grads_k.append(list(grad))

        # === 平均梯度 ===
        avg_grad_list = [torch.mean(torch.stack([grads[j] for grads in prompt_grads_k], dim=0), dim=0)
                         for j in range(len(prompt_grads_k[0]))]

        # === 计算每个 prompt 的 score + EMA ===
        final_prompt_grads = []
        for i in range(self.model.num_prompts):
            similarity = F.cosine_similarity(prompt_grads_k[i][0].flatten(),
                                             avg_grad_list[0].flatten(), dim=0)
            consistency_score = (1 + similarity) / 2
            performance_score = torch.exp(-prompt_losses[i].detach())
            grad_norm = torch.norm(torch.cat([g.flatten() for g in prompt_grads_k[i]]))
            magnitude_score = grad_norm / (grad_norm + 1e-8)

            current_score = consistency_score * performance_score * magnitude_score
            with torch.no_grad():
                self.model.prompt_ema_scores[i] = (
                        self.ema_beta * self.model.prompt_ema_scores[i]
                        + (1 - self.ema_beta) * current_score
                )

            weighted_grad = [g * current_score for g in prompt_grads_k[i]]
            final_prompt_grads.append(weighted_grad)

        # === 手动赋值梯度 ===
        for i in range(self.model.num_prompts):
            for param, grad in zip(self.model.prompt_ensemble[i].parameters(), final_prompt_grads[i]):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad = param.grad + grad

        # === 参数更新 ===
        self.optimizer_backbone.step()
        self.optimizer_prompts.step()

        return {
            'loss': {
                'total': final_loss_for_backbone.item(),
                'position': pos_loss.item(),
                'velocity': vel_loss.item()
            }
        }

    def inference_step(self, batch, epoch, batch_step, mode, data_loader=None):
        """Execute inference step"""
        self.model.eval()
        with torch.no_grad():
            criterion = TrajectoryLoss(loss_type='mse')

            targets, _ = create_fine_grained_target(
                batch['decoder'],
                num_balls=getattr(data_loader, 'n_balls', self.model.num_balls),
                time_length=self.model.prediction_len
            )

            model_output = self.model(batch, mode='test')
            predictions_group, _, _ = model_output

            # === 按 num_prompts 聚合 ===
            # predictions_group = []
            # for i in range(self.model.num_prompts):
            #     predictions = predictions_list[i::self.model.num_prompts]
            #     predictions_group.append(torch.stack(predictions, dim=0).permute(1, 2, 0, 3))

            predictions_stack = torch.stack(predictions_group, dim=0)

            # === softmax 权重 ===
            weights = F.softmax(self.model.prompt_ema_scores / self.inference_temperature, dim=0).to(self.device)
            weighted_predictions_avg = torch.sum(predictions_stack * weights.view(-1, 1, 1, 1, 1), dim=0)
            batch_data = batch['encoder'].x.view(batch['encoder'].batch_size, 10, -1, 4).to(self.device)
            #filter_and_save_samples(weighted_predictions_avg, targets, batch_data, mode='test')
            # === 加权平均 ===

            total_loss, pos_loss, vel_loss, pos_mse_x, pos_mse_y, vel_mse_x, vel_mse_y = criterion(weighted_predictions_avg, targets)



            return total_loss, pos_loss, vel_loss, pos_mse_x, pos_mse_y, vel_mse_x, vel_mse_y
