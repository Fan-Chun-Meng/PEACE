import logging

from lib.DS_ODE import DS_ODE


def create_enhanced_promptode(device, args):

    logger = logging.getLogger(__name__)
    logger.info("Starting to create EnhancedPromptODE model")

    config = {
        'batch_size': args.batch_size,
        'num_particles': args.n_balls,
        'history_len': args.input_length-1,
        'prediction_len': args.pred_length,  # 我们要预测未来 7 个时间步
        'pos_dim': 2,
        'state_dim': 4,
        'param_dim': 4,
        'hidden_dim': 64,
        'num_prompts':args.num_prompts,
        'tf_rate':0.3
    }

    # 3. 实例化完整模型和损失函数
    model = DS_ODE(config).to(device)
    logger.info("Successfully created EnhancedPromptODE model basic structure")

    
    return model
