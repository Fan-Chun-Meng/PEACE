import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
num_particles = 10  # 假设粒子数是固定的


def filter_and_save_samples(pre, tar, hist=None, threshold=0.01, save_list=None, mode="train"):
    """
    从 batch 中筛选 MAE < threshold 的样本，并保存下来，附带历史轨迹 hist。

    参数:
    - pre: [B, N, T, 4] 预测
    - tar: [B, N, T, 4] 真实
    - hist: [B, N, T_hist, 4] 历史轨迹，可为 None
    - threshold: MAE 阈值
    - save_list: 用于保存的 list
    - mode: "train" 或 "test"，用于区分文件名或保存标识

    返回:
    - save_list: 更新后的保存列表
    """
    if save_list is None:
        save_list = []

    # 计算 batch 内每个样本的 MAE（仅 pos 或全部特征）
    mae_per_sample = torch.mean(torch.abs(pre - tar), dim=(1, 2, 3))

    # 找出误差小于阈值的样本索引
    good_idx = (mae_per_sample < threshold).nonzero(as_tuple=True)[0]

    if len(good_idx) > 0:
        good_pre = pre[good_idx]
        good_tar = tar[good_idx]
        good_hist = hist[good_idx] if hist is not None else None

        for i in range(len(good_idx)):
            item = {
                "pre": good_pre[i].detach().cpu(),
                "tar": good_tar[i].detach().cpu(),
                "mae": mae_per_sample[good_idx[i]].item(),
                "mode": mode
            }
            if good_hist is not None:
                item["hist"] = good_hist[i].detach().cpu()
            save_list.append(item)

    return save_list

def smooth_trajectory(x, y, method="savgol", window=7, polyorder=2, interp_points=200):
    """
    对轨迹 (x, y) 进行平滑
    """
    if method == "moving":
        def moving_average(data, w):
            return np.convolve(data, np.ones(w)/w, mode="valid")
        min_len = min(len(x), len(y))
        x_smooth = moving_average(x[:min_len], window)
        y_smooth = moving_average(y[:min_len], window)

    elif method == "savgol":
        window = min(len(x) // 2 * 2 - 1, window) if len(x) > 5 else len(x)
        if window < 3:
            return x, y
        x_smooth = savgol_filter(x, window_length=window, polyorder=polyorder)
        y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)

    elif method == "spline":
        t = np.arange(len(x))
        t_new = np.linspace(0, len(x) - 1, interp_points)
        x_smooth = make_interp_spline(t, x, k=3)(t_new)
        y_smooth = make_interp_spline(t, y, k=3)(t_new)

    else:
        raise ValueError("Unsupported smoothing method")

    return x_smooth, y_smooth


# ==============================
# 绘制轨迹函数
# ==============================
def draw_picture(history_tensor,
                 future_tensor,
                 mode,
                 epoch,
                 batch_step,
                 save_path,
                 num_particles=10,
                 smooth_method="savgol",
                 xlim=None,
                 ylim=None):
    """
    可视化历史轨迹和未来轨迹，并支持轨迹平滑
    - history_tensor: [batch, N, T_obs, 4]
    - future_tensor:  [batch, N, T_pred, 4]
    - xlim: (xmin, xmax)，设置横坐标范围
    - ylim: (ymin, ymax)，设置纵坐标范围
    """
    # --- 数据准备 ---
    if isinstance(history_tensor, torch.Tensor):
        history_np = history_tensor.detach().cpu().numpy()
    else:
        history_np = history_tensor

    if isinstance(future_tensor, torch.Tensor):
        future_np = future_tensor.detach().cpu().numpy()
    else:
        future_np = future_tensor

    # --- 批量可视化 ---
    for batch_index in range(0, history_tensor.shape[0]):  # 这里只画一个batch
        if batch_index >= history_np.shape[0]:
            continue

        history_batch = history_np[batch_index, :, :, :]
        future_batch = future_np[batch_index, :, :, :]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.jet(np.linspace(0, 1, num_particles))

        for i in range(num_particles):
            hist_qx = history_batch[i, :, 0]
            hist_qy = history_batch[i, :, 1]
            future_qx = future_batch[i, :, 0]
            future_qy = future_batch[i, :, 1]

            # 历史轨迹
            ax.plot(hist_qx, hist_qy,
                    linestyle='--', alpha=0.6, color=colors[i])

            # 拼接历史末点和未来轨迹
            plot_future_qx = np.concatenate([hist_qx[-1:], future_qx])
            plot_future_qy = np.concatenate([hist_qy[-1:], future_qy])

            # 平滑未来轨迹
            plot_future_qx, plot_future_qy = smooth_trajectory(
                plot_future_qx, plot_future_qy, method=smooth_method
            )

            # 绘制平滑后的未来轨迹
            ax.plot(plot_future_qx, plot_future_qy,
                    marker='o', linestyle='-', label=f'Particle {i + 1}',
                    color=colors[i], markersize=3)

            # 起点
            ax.plot(hist_qx[0], hist_qy[0], 'x', color=colors[i],
                    markersize=10, markeredgewidth=2)

        # 标题和标签
        ax.set_title(f'Trajectory [{mode}] - Epoch {epoch}, Batch {batch_step}, Sample {batch_index}')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Position y')
        ax.grid(True)
        ax.axis('equal')

        # === 设置坐标范围 ===
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        # plt.show()
        filename = str(batch_index)+'.png'
        plt.savefig(save_path+'/'+filename)
        plt.close()