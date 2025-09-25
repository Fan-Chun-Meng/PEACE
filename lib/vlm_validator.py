import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

class TrajectoryVisualizer:
    """轨迹可视化器"""
    def __init__(self, figsize=(8, 6), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def create_trajectory_plot(self, positions, velocities=None, title="Physics Trajectory"):
        """
        创建轨迹图
        positions: [time_steps, num_particles, 2] 位置数据
        velocities: [time_steps, num_particles, 2] 速度数据（可选）
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        time_steps, num_particles, _ = positions.shape
        
        # 绘制每个粒子的轨迹
        for particle_idx in range(num_particles):
            particle_pos = positions[:, particle_idx, :]
            
            # 轨迹线
            ax.plot(particle_pos[:, 0], particle_pos[:, 1], 
                   color=self.colors[particle_idx % len(self.colors)],
                   alpha=0.7, linewidth=2, 
                   label=f'Particle {particle_idx}')
            
            # 起点和终点标记
            ax.scatter(particle_pos[0, 0], particle_pos[0, 1], 
                      color=self.colors[particle_idx % len(self.colors)],
                      s=100, marker='o', edgecolors='black', linewidth=2)
            ax.scatter(particle_pos[-1, 0], particle_pos[-1, 1],
                      color=self.colors[particle_idx % len(self.colors)],
                      s=100, marker='s', edgecolors='black', linewidth=2)
            
            # 可选：速度箭头
            if velocities is not None:
                # 每隔几个时间步显示速度箭头
                step = max(1, time_steps // 5)
                for t in range(0, time_steps, step):
                    pos = particle_pos[t]
                    vel = velocities[t, particle_idx]
                    ax.arrow(pos[0], pos[1], vel[0]*0.1, vel[1]*0.1,
                            head_width=0.05, head_length=0.05,
                            fc=self.colors[particle_idx % len(self.colors)],
                            ec=self.colors[particle_idx % len(self.colors)],
                            alpha=0.6)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 设置坐标轴范围
        all_pos = positions.reshape(-1, 2)
        margin = 0.1
        x_range = all_pos[:, 0].max() - all_pos[:, 0].min()
        y_range = all_pos[:, 1].max() - all_pos[:, 1].min()
        
        ax.set_xlim(all_pos[:, 0].min() - margin * x_range,
                   all_pos[:, 0].max() + margin * x_range)
        ax.set_ylim(all_pos[:, 1].min() - margin * y_range,
                   all_pos[:, 1].max() + margin * y_range)
        
        plt.tight_layout()
        return fig
    
    def create_phase_space_plot(self, positions, velocities, title="Phase Space"):
        """创建相空间图"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        num_particles = positions.shape[1]
        
        for particle_idx in range(min(num_particles, 4)):
            row, col = particle_idx // 2, particle_idx % 2
            ax = axes[row, col]
            
            pos = positions[:, particle_idx, :]
            vel = velocities[:, particle_idx, :]
            
            # X方向相空间
            ax.plot(pos[:, 0], vel[:, 0], 'b-', alpha=0.7, label='X-direction')
            # Y方向相空间
            ax.plot(pos[:, 1], vel[:, 1], 'r-', alpha=0.7, label='Y-direction')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Velocity')
            ax.set_title(f'Particle {particle_idx} Phase Space')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def fig_to_image(self, fig):
        """将matplotlib图转换为PIL Image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return image


class VLMValidator:
    """基于Vision-Language Model的物理合理性验证器"""
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cuda"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.visualizer = TrajectoryVisualizer()
        
        # 物理合理性检查的关键词
        self.physics_keywords = {
            'valid': ['smooth', 'continuous', 'trajectory', 'motion', 'path', 'movement', 'flow'],
            'invalid': ['discontinuous', 'jump', 'teleport', 'break', 'error', 'noise', 'random']
        }
    
    def validate_trajectory(self, predicted_positions, predicted_velocities=None, 
                          domain_type="spring", confidence_threshold=0.7):
        """
        验证预测轨迹的物理合理性
        
        Args:
            predicted_positions: [time_steps, num_particles, 2]
            predicted_velocities: [time_steps, num_particles, 2] 
            domain_type: "spring" or "charged"
            confidence_threshold: 置信度阈值
            
        Returns:
            dict: 验证结果
        """
        
        # 1. 创建轨迹可视化
        trajectory_fig = self.visualizer.create_trajectory_plot(
            predicted_positions, predicted_velocities,
            title=f"{domain_type.capitalize()} System Trajectory"
        )
        trajectory_image = self.visualizer.fig_to_image(trajectory_fig)
        
        # 2. VLM描述生成
        description = self.generate_description(trajectory_image)
        
        # 3. 物理合理性评估
        physics_score = self.evaluate_physics_consistency(
            description, predicted_positions, domain_type
        )
        
        # 4. 额外的数值检查
        numerical_score = self.numerical_physics_check(
            predicted_positions, predicted_velocities, domain_type
        )
        
        # 5. 综合评估
        final_score = 0.6 * physics_score + 0.4 * numerical_score
        is_valid = final_score > confidence_threshold
        
        return {
            'is_valid': is_valid,
            'final_score': final_score,
            'physics_score': physics_score,
            'numerical_score': numerical_score,
            'description': description,
            'trajectory_image': trajectory_image,
            'domain_type': domain_type
        }
    
    def generate_description(self, image):
        """使用VLM生成图像描述"""
        try:
            # 准备输入
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
            
            description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return description
            
        except Exception as e:
            print(f"VLM description generation failed: {e}")
            return "Description generation failed"
    
    def evaluate_physics_consistency(self, description, positions, domain_type):
        """基于描述评估物理一致性"""
        description_lower = description.lower()
        
        # 计算关键词得分
        valid_score = sum(1 for word in self.physics_keywords['valid'] 
                         if word in description_lower)
        invalid_score = sum(1 for word in self.physics_keywords['invalid'] 
                           if word in description_lower)
        
        # 基础得分
        base_score = (valid_score - invalid_score) / max(valid_score + invalid_score, 1)
        base_score = max(0, min(1, (base_score + 1) / 2))  # 归一化到[0,1]
        
        # Domain-specific检查
        domain_bonus = 0
        if domain_type == "spring":
            if any(word in description_lower for word in ['spring', 'oscillat', 'bounce', 'elastic']):
                domain_bonus += 0.2
        elif domain_type == "charged":
            if any(word in description_lower for word in ['attract', 'repel', 'charge', 'electric']):
                domain_bonus += 0.2
        
        final_score = min(1.0, base_score + domain_bonus)
        return final_score
    
    def numerical_physics_check(self, positions, velocities, domain_type):
        """数值物理检查"""
        score = 0.0
        
        # 1. 连续性检查
        continuity_score = self.check_continuity(positions)
        score += 0.3 * continuity_score
        
        # 2. 边界条件检查
        boundary_score = self.check_boundaries(positions)
        score += 0.2 * boundary_score
        
        # 3. 速度合理性检查
        if velocities is not None:
            velocity_score = self.check_velocity_reasonableness(velocities)
            score += 0.3 * velocity_score
        
        # 4. Domain-specific检查
        if domain_type == "spring":
            spring_score = self.check_spring_physics(positions, velocities)
            score += 0.2 * spring_score
        elif domain_type == "charged":
            charge_score = self.check_charge_physics(positions, velocities)
            score += 0.2 * charge_score
        
        return min(1.0, score)
    
    def check_continuity(self, positions):
        """检查位置连续性"""
        # 计算相邻时间步的位置差
        position_diffs = np.diff(positions, axis=0)
        max_displacement = np.max(np.linalg.norm(position_diffs, axis=-1))
        
        # 合理的最大位移阈值
        reasonable_threshold = 1.0
        continuity_score = max(0, 1 - max_displacement / reasonable_threshold)
        
        return continuity_score
    
    def check_boundaries(self, positions):
        """检查边界条件"""
        # 假设仿真边界为[-5, 5]
        boundary = 5.0
        
        # 检查是否有粒子超出边界
        out_of_bounds = np.any((np.abs(positions) > boundary * 1.1))
        
        if out_of_bounds:
            return 0.5  # 轻微扣分
        return 1.0
    
    def check_velocity_reasonableness(self, velocities):
        """检查速度合理性"""
        # 检查速度是否过大
        max_velocity = np.max(np.linalg.norm(velocities, axis=-1))
        reasonable_max_vel = 2.0
        
        if max_velocity > reasonable_max_vel:
            return max(0, 1 - (max_velocity - reasonable_max_vel) / reasonable_max_vel)
        return 1.0
    
    def check_spring_physics(self, positions, velocities):
        """检查弹簧物理"""
        # 简化检查：粒子间距离不应该剧烈变化
        if positions.shape[1] < 2:
            return 1.0
        
        distances = []
        for t in range(positions.shape[0]):
            for i in range(positions.shape[1]):
                for j in range(i+1, positions.shape[1]):
                    dist = np.linalg.norm(positions[t, i] - positions[t, j])
                    distances.append(dist)
        
        # 距离变化应该相对平滑
        distances = np.array(distances).reshape(positions.shape[0], -1)
        distance_variations = np.std(distances, axis=0)
        avg_variation = np.mean(distance_variations)
        
        # 合理的变化阈值
        reasonable_variation = 1.0
        spring_score = max(0, 1 - avg_variation / reasonable_variation)
        
        return spring_score
    
    def check_charge_physics(self, positions, velocities):
        """检查带电粒子物理"""
        # 简化实现
        return 1.0


class DataAugmentationManager:
    """数据增强管理器"""
    def __init__(self, validator, max_augmented_samples=1000):
        self.validator = validator
        self.max_augmented_samples = max_augmented_samples
        self.augmented_data = []
        self.validation_history = []
    
    def validate_and_augment(self, predictions, targets, domain_type, confidence_threshold=0.7):
        """验证预测并进行数据增强"""
        batch_size = predictions.shape[0]
        valid_samples = []
        
        for i in range(batch_size):
            # 提取单个样本的预测
            pred_positions = predictions[i].detach().cpu().numpy()
            target_positions = targets[i].detach().cpu().numpy()
            
            # 验证预测的物理合理性
            validation_result = self.validator.validate_trajectory(
                pred_positions, domain_type=domain_type,
                confidence_threshold=confidence_threshold
            )
            
            self.validation_history.append({
                'score': validation_result['final_score'],
                'is_valid': validation_result['is_valid'],
                'domain_type': domain_type
            })
            
            # 如果预测通过验证，添加为训练数据
            if validation_result['is_valid']:
                augmented_sample = {
                    'input': target_positions[:-1],  # 去掉最后一步作为输入
                    'target': pred_positions[-1:],   # 预测的最后一步作为标签
                    'domain_type': domain_type,
                    'confidence': validation_result['final_score']
                }
                valid_samples.append(augmented_sample)
        
        # 添加到增强数据池
        self.augmented_data.extend(valid_samples)
        
        # 保持数据池大小限制
        if len(self.augmented_data) > self.max_augmented_samples:
            # 保留置信度最高的样本
            self.augmented_data.sort(key=lambda x: x['confidence'], reverse=True)
            self.augmented_data = self.augmented_data[:self.max_augmented_samples]
        
        return len(valid_samples), validation_result['final_score']
    
    def get_augmented_batch(self, batch_size):
        """获取增强数据批次"""
        if len(self.augmented_data) < batch_size:
            return None
        
        # 随机采样
        indices = np.random.choice(len(self.augmented_data), batch_size, replace=False)
        batch = [self.augmented_data[i] for i in indices]
        
        return batch
    
    def get_validation_statistics(self):
        """获取验证统计信息"""
        if not self.validation_history:
            return {}
        
        recent_history = self.validation_history[-100:]  # 最近100个样本
        
        return {
            'total_validated': len(self.validation_history),
            'recent_valid_rate': np.mean([h['is_valid'] for h in recent_history]),
            'recent_avg_score': np.mean([h['score'] for h in recent_history]),
            'augmented_samples': len(self.augmented_data)
        } 