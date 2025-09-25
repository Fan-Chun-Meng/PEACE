#!/usr/bin/env python3

import torch
import os
import sys
from datetime import datetime

from lib.update_dataLoader import update_dataLoader

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modularized components
from lib.config_parser import parse_arguments, setup_logger
from lib.data_utils import setup_data_loaders, validate_data_loaders
from lib.model_utils import create_enhanced_promptode
from lib.integrated_system import IntegratedPromptODESystem
from lib.training_utils import train_epoch, validate_epoch, test_model


def main():
    for condition_num in [12]:
        for time in [12,24,36]:
            for num_prompts in [3]:
                # Parse arguments and setup
                args = parse_arguments()

                args.input_length = condition_num
                args.pred_length = time
                args.num_prompts = num_prompts
                args.extrap_num = time
                # Setup device and logging
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logger = setup_logger()

                logger.info("=" * 80)
                logger.info("Domain Generalization Framework Training Started")
                logger.info("=" * 80)
                logger.info(f"Device: {device}")
                logger.info(f"Input length: {args.input_length}")
                logger.info(f"Prediction length: {args.pred_length}")
                logger.info(f"Training epochs: {args.epochs}")
                logger.info(f"Using 12-step prediction configuration")


                logger.info("[SUCCESS] PGODE model loaded successfully")
                logger.info("Creating DG PromptODE model...")
                model = create_enhanced_promptode(device, args)
                logger.info("[SUCCESS] DG model created successfully")

                logger.info("   - Lower branch: Fixed feature extractor")
                logger.info("   - Upper branch: Dynamic prompt evolution")

                # Setup data loaders
                data_loaders, original_data_loader = setup_data_loaders(args, logger)

                if not validate_data_loaders(data_loaders, logger):
                    logger.error("[ERROR] Data loader validation failed")
                    exit(1)

                # Create integrated training system
                logger.info("Creating integrated training system...")
                system = IntegratedPromptODESystem(
                    model=model,
                    device=device,
                    args=args
                )
                logger.info("[SUCCESS] Integrated system created successfully")

                # Training loop
                logger.info(f"Starting DG training - {args.epochs} epochs...")
                best_val_loss = float('inf')
                patience_counter = 0

                for epoch in range(args.epochs):
                    logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
                    logger.info("-" * 50)

                    # Training
                    train_losses = train_epoch(system, data_loaders, device, logger, epoch, original_data_loader)
                    if epoch != 0 and epoch % 40 == 0:
                        data_loaders = update_dataLoader(data_loaders, device,original_data_loader,train_losses,args)
                    # Validation
                    val_losses = validate_epoch(system, data_loaders, device, logger, epoch, args, original_data_loader)

                    test_results = test_model(system, data_loaders, device, logger, epoch, args, original_data_loader)

                    logger.info(f"Epoch {epoch + 1}:")
                    logger.info(f"  Training loss: Total={train_losses['total']:.6f}, "
                                f"Position={train_losses['position']:.6f}, "
                                f"Velocity={train_losses['velocity']:.6f}")
                    logger.info(f"  Validation loss: Total={val_losses['total']:.6f}, "
                                f"Position={val_losses['position']:.6f}, "
                                f"Velocity={val_losses['velocity']:.6f}")
                    logger.info(f"ID Domain - Total Loss: {test_results.get('id_total', {}):.6f},"
                                f"ID Domain - Position Loss: {test_results.get('id_position', {}):.6f},"
                                f"ID Domain - Velocity Loss: {test_results.get('id_velocity', {}):.6f},"
                                f"OOD Domain - Total Loss: {test_results.get('ood_total', {}):.6f},"
                                f"OOD Domain - Position Loss: {test_results.get('ood_position', {}):.6f},"
                                f"OOD Domain - Velocity Loss: {test_results.get('ood_velocity', {}):.6f}")
                    # Save best model (using total loss as metric)
                    if val_losses['total'] < best_val_loss:
                        best_val_loss = val_losses['total']
                        patience_counter = 0

                        # Save model
                        os.makedirs(args.save_path, exist_ok=True)
                        save_file = os.path.join(args.save_path, f'best_dg_model_epoch_{epoch + 1}.pth')
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': system.model.state_dict(),
                            'val_loss': val_losses['total'],
                            'args': args
                        }, save_file)
                        logger.info(f"[SAVE] Best model saved: {save_file}")
                    else:
                        patience_counter += 1

                    # Early stopping
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                        break

                # Final testing
                logger.info("\n" + "=" * 80)
                logger.info("FINAL TESTING")
                logger.info("=" * 80)

                test_results = test_model(system, data_loaders, device, logger, args, original_data_loader)

                logger.info("\nFinal Test Results:")
                logger.info(f"ID Domain - Total Loss: {test_results.get('id_total', {}):.6f}")
                logger.info(f"ID Domain - Position Loss: {test_results.get('id_position', {}):.6f}")
                logger.info(f"ID Domain - Velocity Loss: {test_results.get('id_velocity', {}):.6f}")
                logger.info(f"OOD Domain - Total Loss: {test_results.get('ood_total', {}):.6f}")
                logger.info(f"OOD Domain - Position Loss: {test_results.get('ood_position', {}):.6f}")
                logger.info(f"OOD Domain - Velocity Loss: {test_results.get('ood_velocity', {}):.6f}")

                logger.info("\n" + "=" * 80)
                logger.info("TRAINING COMPLETED SUCCESSFULLY")
                logger.info("=" * 80)


if __name__ == '__main__':
    main()