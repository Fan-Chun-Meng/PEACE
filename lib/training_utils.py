"""Training Utilities

This module contains utility functions for training, validation, and testing
of the PromptODE model, including epoch-level training loops and model evaluation.
"""

import torch



def train_epoch(system, data_loaders, device, logger, epoch, original_data_loader=None):

    # Check if training data is available
    if 'train' not in data_loaders:
        logger.error("No training data available")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}
    
    epoch_losses = []
    
    for i in range(data_loaders['train']['num_batch']):
        # Get a batch of data
        encoder_batch = next(data_loaders['train']['encoder'])
        decoder_batch = next(data_loaders['train']['decoder'])
        graph_batch = next(data_loaders['train']['graph'])
        sys_para_batch = next(data_loaders['train']['sys_para'])

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

        # Execute training step - pass original data_loader instance to get global_norm_params
        result = system.train_step(batch, original_data_loader, i, epoch)
        epoch_losses.append(result['loss'])

        # Output loss only at the end of each batch
        logger.info(f"Batch {i}: Total Loss={result['loss']['total']:.6f}, "
                  f"Position Loss={result['loss']['position']:.6f}, "
                  f"Velocity Loss={result['loss']['velocity']:.6f}")
    
    if not epoch_losses:
        logger.warning("No valid batches processed during training")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}
    
    avg_losses = {
        'total': sum(d['total'] for d in epoch_losses) / len(epoch_losses),
        'position': sum(d['position'] for d in epoch_losses) / len(epoch_losses),
        'velocity': sum(d['velocity'] for d in epoch_losses) / len(epoch_losses)
    }
    
    logger.info(f"Training completed, average losses: Total Loss={avg_losses['total']:.6f}, "
              f"Position Loss={avg_losses['position']:.6f}, "
              f"Velocity Loss={avg_losses['velocity']:.6f}")
    
    return avg_losses


def validate_epoch(system, data_loaders, device, logger, epoch, args, original_data_loader=None):

    epoch_losses = []
    epoch_pos = []
    epoch_vel = []
    epoch_pos_x = []
    epoch_pos_y = []
    epoch_vel_x = []
    epoch_vel_y = []
    # Check validation data loader validity
    if 'val' not in data_loaders or not data_loaders['val']:
        logger.error("No validation data available")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}

    
    for i in range(data_loaders['val']['num_batch']):
        # Create batch dictionary
        batch = {}

        # Get a batch of data and move to device
        batch['encoder'] = next(data_loaders['val']['encoder']).to(device)
        batch['decoder'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in next(data_loaders['val']['decoder']).items()}
        batch['graph'] = next(data_loaders['val']['graph']).to(device)
        batch['sys_para'] = next(data_loaders['val']['sys_para']).to(device)

        # Execute inference step
        result = system.inference_step(batch,epoch, i, 'val', original_data_loader)
        epoch_losses.append(result[0])
        epoch_pos.append(result[1])
        epoch_vel.append(result[2])
        epoch_pos_x.append(result[3])
        epoch_pos_y.append(result[4])
        epoch_vel_x.append(result[5])
        epoch_vel_y.append(result[6])
        # Output loss only at the end of each batch
        logger.info(f"Batch {i}: Total Loss={result[0]:.6f}, "
                    f"Position Loss={result[1]:.6f}, "
                    f"Velocity Loss={result[2]:.6f}")

    if not epoch_losses:
        logger.warning("No valid batches processed during val")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}

    avg_losses = {
        'total': sum(epoch_losses) / len(epoch_losses),
        'position': sum(epoch_pos) / len(epoch_pos),
        'velocity': sum(epoch_vel) / len(epoch_vel),
        'position_pos_x': sum(epoch_pos_x) / len(epoch_pos_x),
        'position_pos_y': sum(epoch_pos_y) / len(epoch_pos_y),
        'velocity_pos_x': sum(epoch_vel_x) / len(epoch_vel_x),
        'velocity_pos_y': sum(epoch_vel_y) / len(epoch_vel_y)
    }

    logger.info(f"Val completed, average losses: Total Loss={avg_losses['total']:.6f}, "
                f"Position Loss={avg_losses['position']:.6f}, Position X={avg_losses['position_pos_x']:.6f}, Position Y={avg_losses['position_pos_y']:.6f}"
                f"Velocity Loss={avg_losses['velocity']:.6f}, Velocity X={avg_losses['velocity_pos_x']:.6f}, Velocity Y={avg_losses['velocity_pos_y']:.6f}")

    return {'total': avg_losses['total'], 'position': avg_losses['position'],'velocity': avg_losses['velocity']}


def test_model(system, data_loaders, device, logger, epoch, args, original_data_loader=None):

    epoch_losses = []
    epoch_pos = []
    epoch_vel = []
    epoch_pos_x = []
    epoch_pos_y = []
    epoch_vel_x = []
    epoch_vel_y = []
    # Check validation data loader validity
    if 'id_test' not in data_loaders or not data_loaders['id_test']:
        logger.error("No id_test data available")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}

    for i in range(data_loaders['id_test']['num_batch']):
        # Create batch dictionary
        batch = {}

        # Get a batch of data and move to device
        batch['encoder'] = next(data_loaders['id_test']['encoder']).to(device)
        batch['decoder'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in next(data_loaders['id_test']['decoder']).items()}
        batch['graph'] = next(data_loaders['id_test']['graph']).to(device)
        batch['sys_para'] = next(data_loaders['id_test']['sys_para']).to(device)

        # Execute inference step
        result = system.inference_step(batch, epoch, i, 'id',original_data_loader)
        epoch_losses.append(result[0])
        epoch_pos.append(result[1])
        epoch_vel.append(result[2])
        epoch_pos_x.append(result[3])
        epoch_pos_y.append(result[4])
        epoch_vel_x.append(result[5])
        epoch_vel_y.append(result[6])
        # Output loss only at the end of each batch
        # logger.info(f"Batch {i}: Total Loss={result[0]:.6f}, "
        #             f"Position Loss={result[1]:.6f}, "
        #             f"Velocity Loss={result[2]:.6f}")

    if not epoch_losses:
        logger.warning("No id_test batches processed during test")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}

    id_avg_losses = {
        'total': sum(epoch_losses) / len(epoch_losses),
        'position': sum(epoch_pos) / len(epoch_pos),
        'velocity': sum(epoch_vel) / len(epoch_vel),
        'position_pos_x': sum(epoch_pos_x) / len(epoch_pos_x),
        'position_pos_y': sum(epoch_pos_y) / len(epoch_pos_y),
        'velocity_pos_x': sum(epoch_vel_x) / len(epoch_vel_x),
        'velocity_pos_y': sum(epoch_vel_y) / len(epoch_vel_y)
    }

    logger.info(f"id_test completed, average losses: Total Loss={id_avg_losses['total']:.6f}, "
                f"Position Loss={id_avg_losses['position']:.6f}, Position X={id_avg_losses['position_pos_x']:.6f}, Position Y={id_avg_losses['position_pos_y']:.6f}"
                f"Velocity Loss={id_avg_losses['velocity']:.6f}, Velocity X={id_avg_losses['velocity_pos_x']:.6f}, Velocity Y={id_avg_losses['velocity_pos_y']:.6f}")

    epoch_losses = []
    epoch_pos = []
    epoch_vel = []
    epoch_pos_x = []
    epoch_pos_y = []
    epoch_vel_x = []
    epoch_vel_y = []
    # Check validation data loader validity
    if 'ood_test' not in data_loaders or not data_loaders['ood_test']:
        logger.error("No ood_test data available")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}

    for i in range(data_loaders['ood_test']['num_batch']):
        # Create batch dictionary
        batch = {}

        # Get a batch of data and move to device
        batch['encoder'] = next(data_loaders['ood_test']['encoder']).to(device)
        batch['decoder'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in next(data_loaders['ood_test']['decoder']).items()}
        batch['graph'] = next(data_loaders['ood_test']['graph']).to(device)
        batch['sys_para'] = next(data_loaders['ood_test']['sys_para']).to(device)

        # Execute inference step
        result = system.inference_step(batch, epoch, i, 'ood',original_data_loader)
        epoch_losses.append(result[0])
        epoch_pos.append(result[1])
        epoch_vel.append(result[2])
        epoch_pos_x.append(result[3])
        epoch_pos_y.append(result[4])
        epoch_vel_x.append(result[5])
        epoch_vel_y.append(result[6])
        # Output loss only at the end of each batch
        logger.info(f"Batch {i}: Total Loss={result[0]:.6f}, "
                    f"Position Loss={result[1]:.6f}, "
                    f"Velocity Loss={result[2]:.6f}")

    if not epoch_losses:
        logger.warning("No ood_test batches processed during test")
        return {'total': float('inf'), 'position': float('inf'), 'velocity': float('inf')}

    ood_avg_losses = {
        'total': sum(epoch_losses) / len(epoch_losses),
        'position': sum(epoch_pos) / len(epoch_pos),
        'velocity': sum(epoch_vel) / len(epoch_vel),
        'position_pos_x': sum(epoch_pos_x) / len(epoch_pos_x),
        'position_pos_y': sum(epoch_pos_y) / len(epoch_pos_y),
        'velocity_pos_x': sum(epoch_vel_x) / len(epoch_vel_x),
        'velocity_pos_y': sum(epoch_vel_y) / len(epoch_vel_y)
    }

    logger.info(f"ood_test completed, average losses: Total Loss={ood_avg_losses['total']:.6f}, "
                f"Position Loss={ood_avg_losses['position']:.6f}, Position X={ood_avg_losses['position_pos_x']:.6f}, Position Y={ood_avg_losses['position_pos_y']:.6f}"
                f"Velocity Loss={ood_avg_losses['velocity']:.6f}, Velocity X={ood_avg_losses['velocity_pos_x']:.6f}, Velocity Y={ood_avg_losses['velocity_pos_y']:.6f}")

    return {'id_total': id_avg_losses['total'], 'id_position': id_avg_losses['position'], 'id_velocity': id_avg_losses['velocity'],'ood_total': ood_avg_losses['total'], 'ood_position': ood_avg_losses['position'], 'ood_velocity': ood_avg_losses['velocity']}