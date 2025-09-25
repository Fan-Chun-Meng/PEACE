import torch
import argparse
from lib.new_dataLoader import ParseData

def create_data_args(args):
    """Create data loading arguments from main arguments."""
    data_args = argparse.Namespace(
        random_seed=42,
        total_ode_step=24,
        total_ode_step_train=24,
        cutting_edge=True,
        extrap_num=args.extrap_num,
        condition_num=args.input_length,
        mode=args.mode,
        suffix=args.suffix,
        num_domains = args.num_domains
    )
    return data_args

def setup_data_loaders(args, logger):

    logger.info("Loading multi-domain data...")
    
    # Create data loading parameters
    data_args = create_data_args(args)
    
    # Create main data loader (ID domain)
    enable_normalization = not args.disable_norm  # Disable normalization if --disable-norm is specified
    data_loader = ParseData(
        dataset_path=args.id_data_path,  # Use ID data path as main data path
        args=data_args,
        suffix=args.suffix,
        mode="extrap",
        enable_normalization=enable_normalization
    )
    # Load ID domain test data
    id_test_loader = data_loader.load_data(
        sample_percent=0.15,
        batch_size=args.batch_size,
        data_type="test",
        domain_id=0
    )
    # Load training and validation data
    train_loader = data_loader.load_data(
        sample_percent=0.7,
        batch_size=args.batch_size,
        data_type="train",
        domain_id=0
    )



    val_loader = data_loader.load_data(
        sample_percent=0.15,
        batch_size=args.batch_size,
        data_type="val",
        domain_id=0
    )
    
    # Create OOD data loader and load test data
    ood_data_loader = ParseData(
        dataset_path=args.ood_data_path,
        args=data_args,
        suffix=args.suffix,
        mode="extrap"
    )

    ood_test_loader = ood_data_loader.load_data(
        sample_percent=0.15,
        batch_size=args.batch_size,
        data_type="test",
        domain_id=1,
        ood_data_path=args.ood_data_path
    )

    
    # Organize data loaders into dictionary format
    data_loaders = {
        'train': {
            'encoder': train_loader[0],
            'decoder': train_loader[1],
            'graph': train_loader[2],
            'sys_para': train_loader[3],
            'num_batch': train_loader[4]
        },
        'val': {
            'encoder': val_loader[0],
            'decoder': val_loader[1],
            'graph': val_loader[2],
            'sys_para': val_loader[3],
            'num_batch': val_loader[4]
        },
        'id_test': {
            'encoder': id_test_loader[0],  # Use independent ID test data
            'decoder': id_test_loader[1],
            'graph': id_test_loader[2],
            'sys_para': id_test_loader[3],
            'num_batch': id_test_loader[4]
        },
        'ood_test': {
                'encoder': ood_test_loader[0],
                'decoder': ood_test_loader[1],
                'graph': ood_test_loader[2],
                'sys_para': ood_test_loader[3],
                'num_batch': ood_test_loader[4]
        }
    }
    
    logger.info("[SUCCESS] Multi-domain data loading completed")
    
    return data_loaders, data_loader

def prepare_batch_data(encoder_batch, decoder_batch, graph_batch, sys_para_batch, device):

    # Move data to device
    encoder_batch = encoder_batch.to(device)
    decoder_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in decoder_batch.items()}
    graph_batch = graph_batch.to(device)
    sys_para_batch = sys_para_batch.to(device)

    # Combine batch data
    batch = {
        'encoder': encoder_batch,
        'decoder': decoder_batch,
        'graph': graph_batch,
        'sys_para': sys_para_batch
    }
    
    return batch

def get_normalization_params(data_loader):

    norm_params = None
    if hasattr(data_loader, 'enable_normalization') and data_loader.enable_normalization:
        norm_params = data_loader.global_norm_params if hasattr(data_loader, 'global_norm_params') else None
    return norm_params

def validate_data_loaders(data_loaders, logger):

    required_keys = ['train', 'val', 'id_test', 'ood_test']
    
    for key in required_keys:
        if key not in data_loaders:
            logger.error(f"Missing required data loader: {key}")
            return False
            
    # Check test data loaders

        
    logger.info("Data loader validation passed")
    return True