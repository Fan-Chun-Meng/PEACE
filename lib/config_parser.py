import argparse
import logging

def setup_logger():
    """Setup logging configuration."""
    # 获取根logger
    logger = logging.getLogger()
    
    # 清除现有的handlers，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建文件handler
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Enhanced PromptODE Training with Domain Generalization')
    
    # Basic training parameters
    parser.add_argument('--dataset', type=str, default='charged', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Data parameters
    parser.add_argument('--n-balls', type=int, default=10, help='Number of balls')
    parser.add_argument('--input_length', type=int, default=12, help='Input sequence length')
    parser.add_argument('--pred-length', type=int, default=12, help='Prediction sequence length')
    parser.add_argument('--num_prompts', type=int, default=3, help='num_prompts')
    parser.add_argument('--up-lr', type=float, default=5e-5, help='Upper level learning rate')
    parser.add_argument('--low-lr', type=float, default=1e-3, help='Lower level learning rate')
    parser.add_argument('--pretrained_path', type=str, default=r'experiments/experiment_75079_GTrans_spring_1000_10_48_expert5_cond12_interp_epoch_4_mse_0.03235475825411933.ckpt', help='Pretrained model path')
    parser.add_argument('--save-path', type=str, default='checkpoints/', help='Model save path')
    parser.add_argument('--id_data_path', type=str, default='data/charged_1000_10_48', help='ID domain data path')
    parser.add_argument('--ood-data-path', type=str, default='data/charged_1000_10_48_ood', help='OOD domain data path')
    parser.add_argument('--mode', type=str, default='interp', help='Data loading mode: interp or extrap')
    parser.add_argument('--suffix', type=str, default='_charged10', help='Data file suffix')
    parser.add_argument('--disable-norm', action='store_true', help='Disable data normalization')
    parser.add_argument('--num_domains', type=int, default=3, help='Number of domains')
    
    # Model architecture parameters
    parser.add_argument('--input-dim', type=int, default=4, help='Input dimension')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    # parser.add_argument('--num_domains', type=int, default=3, help='Number of domains')
    parser.add_argument('--prediction-mode', type=str, default='fine_grained', help='Prediction mode')

    
    return parser.parse_args()
