"""
Training script for Convolutional Autoencoder on FashionMNIST.

Usage:
    python train.py --config config.json
    python train.py --config config.json --out_path results/experiment1
"""

import argparse
import json
import logging
import os

import pandas as pd
import torch

from data_loader import load_fashion_mnist
from model import ConvAutoencoder, AutoencoderTrainer


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_default_config():
    """Return default configuration."""
    return {
        # Model hyperparameters
        "num_layers": 2,
        "base_channels": 16,
        "kernel_size": 3,
        "latent_dim": 64,
        "dropout": 0.2,
        "use_linear": True,
        
        # Training hyperparameters
        "optimizer": "adam",
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 10,
        
        # Data
        "data_dir": "./data",
        "val_split": 0.2,
        "num_workers": 2,
        "seed": 42
    }


def setup_logging(out_path):
    """Setup logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(out_path, 'training.log'), mode='w'),
        ]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder on FashionMNIST")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--out_path", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    
    # Load configuration
    config = get_default_config()
    if args.config:
        user_config = load_config(args.config)
        config.update(user_config)
    
    # Setup output directory
    os.makedirs(args.out_path, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.out_path)
    
    # Save config used for this run
    with open(os.path.join(args.out_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device setup
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)
    
    # Load data
    logger.info("Loading FashionMNIST dataset...")
    train_loader, val_loader, test_loader = load_fashion_mnist(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_split=config['val_split'],
        seed=config['seed']
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = ConvAutoencoder(
        num_layers=config['num_layers'],
        base_channels=config['base_channels'],
        kernel_size=config['kernel_size'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
        use_linear=config['use_linear'],
        device=device
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model config: {model.get_config()}")
    
    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        optimizer_name=config['optimizer'],
        lr=config['lr'],
        device=device
    )
    
    # Training loop
    logger.info("Starting training...")
    metrics_history = []
    
    for epoch in range(config['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
        
        # Train
        train_metrics = trainer.fit(train_loader)
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        # Combine metrics
        epoch_metrics = {'epoch': epoch + 1}
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(val_metrics)
        metrics_history.append(epoch_metrics)
        
        # Log metrics
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in epoch_metrics.items() if k != 'epoch'])
        logger.info(f"  {metrics_str}")
    
    # Final test evaluation
    logger.info("Running final test evaluation...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test loss: {test_metrics['loss']:.6f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(args.out_path, 'metrics.csv'), index=False)
    logger.info(f"Metrics saved to {os.path.join(args.out_path, 'metrics.csv')}")
    
    # Save model weights
    weights_path = os.path.join(args.out_path, 'weights.pth')
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Model weights saved to {weights_path}")
    
    # Save test results
    test_results = {
        'test_loss': test_metrics['loss'],
        'final_train_loss': metrics_history[-1]['train_loss'],
        'final_val_loss': metrics_history[-1]['val_loss']
    }
    with open(os.path.join(args.out_path, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Final train loss: {metrics_history[-1]['train_loss']:.6f}")
    logger.info(f"Final val loss: {metrics_history[-1]['val_loss']:.6f}")
    logger.info(f"Final test loss: {test_metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
