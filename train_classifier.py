"""
Training script for FashionMNIST Classifier with Encoder as feature extractor.

Supports 4 training variants:
    --variant 1: Encoder PRETRAINED + FROZEN, train classifier only
    --variant 2: Encoder NOT PRETRAINED + NOT FROZEN, train end-to-end from scratch
    --variant 3: Encoder PRETRAINED + NOT FROZEN, fine-tune both
    --variant 4: Encoder NOT PRETRAINED + FROZEN, train classifier only

Usage:
    # Variant 1: Pretrained frozen encoder
    python train_classifier.py --variant 1 --encoder_path results/refactor/encoder.pth --encoder_config results/refactor/encoder_config.json --out_path results/classifier_v1

    # Variant 2: Train from scratch
    python train_classifier.py --variant 2 --encoder_config results/refactor/encoder_config.json --out_path results/classifier_v2

    # Variant 3: Fine-tune pretrained
    python train_classifier.py --variant 3 --encoder_path results/refactor/encoder.pth --encoder_config results/refactor/encoder_config.json --out_path results/classifier_v3

    # Variant 4: Random frozen encoder, train classifier only
    python train_classifier.py --variant 4 --encoder_config results/refactor/encoder_config.json --out_path results/classifier_v4
"""

import argparse
import json
import logging
import os

import pandas as pd
import torch

from data_loader import load_fashion_mnist, CLASS_NAMES
from autoencoder import Encoder
from classifier import (
    FashionMNISTClassifier,
    ClassifierTrainer,
    create_classifier_from_pretrained,
    create_classifier_from_scratch,
)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_default_config():
    """Return default configuration."""
    return {
        # Classifier head hyperparameters
        "classifier_conv1_channels": 32,
        "classifier_conv2_channels": 64,
        "classifier_kernel_size": 3,
        "classifier_dropout": 0.3,
        "num_classes": 10,
        
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


def get_variant_description(variant):
    """Return human-readable description of training variant."""
    descriptions = {
        1: "Encoder PRETRAINED + FROZEN, train classifier only",
        2: "Encoder NOT PRETRAINED + NOT FROZEN, train end-to-end from scratch",
        3: "Encoder PRETRAINED + NOT FROZEN, fine-tune both",
        4: "Encoder NOT PRETRAINED + FROZEN, train classifier only"
    }
    return descriptions.get(variant, "Unknown variant")


def main():
    parser = argparse.ArgumentParser(description="Train FashionMNIST Classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--out_path", type=str, default="results/classifier", help="Output directory for results")
    parser.add_argument("--variant", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Training variant: 1=pretrained+frozen, 2=scratch, 3=pretrained+finetune, 4=no_training")
    parser.add_argument("--encoder_path", type=str, default=None, 
                        help="Path to pretrained encoder weights (required for variants 1, 3, 4)")
    parser.add_argument("--encoder_config", type=str, default=None,
                        help="Path to encoder config JSON (required for variants 1, 3, 4)")
    args = parser.parse_args()
    
    # Validate variant requirements
    if args.variant in [1, 3]:
        if not args.encoder_path or not args.encoder_config:
            parser.error(f"Variant {args.variant} requires --encoder_path and --encoder_config")
    
    # Variants 2 and 4 need encoder_config for architecture but not weights
    if args.variant in [2, 4] and not args.encoder_config:
        parser.error(f"Variant {args.variant} requires --encoder_config (for encoder architecture)")
    
    # Load configuration
    config = get_default_config()
    if args.config:
        user_config = load_config(args.config)
        config.update(user_config)
    
    # Setup output directory
    os.makedirs(args.out_path, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.out_path)
    
    # Device setup
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")
    
    # Log variant info
    logger.info("=" * 60)
    logger.info(f"VARIANT {args.variant}: {get_variant_description(args.variant)}")
    logger.info("=" * 60)
    
    # Save config used for this run
    run_config = config.copy()
    run_config['variant'] = args.variant
    run_config['variant_description'] = get_variant_description(args.variant)
    run_config['encoder_path'] = args.encoder_path
    run_config['encoder_config_path'] = args.encoder_config
    
    with open(os.path.join(args.out_path, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
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
    
    # Create model based on variant
    logger.info("Creating model...")
    
    if args.variant == 1:
        # Pretrained + frozen encoder
        model = create_classifier_from_pretrained(
            encoder_weights_path=args.encoder_path,
            encoder_config_path=args.encoder_config,
            encoder_frozen=True,
            conv1_channels=config['classifier_conv1_channels'],
            conv2_channels=config['classifier_conv2_channels'],
            kernel_size=config['classifier_kernel_size'],
            num_classes=config['num_classes'],
            dropout=config['classifier_dropout'],
            device=device
        )
        
    elif args.variant == 2:
        # Fresh encoder, train from scratch
        with open(args.encoder_config, 'r') as f:
            encoder_config = json.load(f)
        encoder_config.pop('output_dim', None)
        
        model = create_classifier_from_scratch(
            encoder_config=encoder_config,
            encoder_frozen=False,
            conv1_channels=config['classifier_conv1_channels'],
            conv2_channels=config['classifier_conv2_channels'],
            kernel_size=config['classifier_kernel_size'],
            num_classes=config['num_classes'],
            dropout=config['classifier_dropout'],
            device=device
        )
        
    elif args.variant == 3:
        # Pretrained + unfrozen encoder (fine-tuning)
        model = create_classifier_from_pretrained(
            encoder_weights_path=args.encoder_path,
            encoder_config_path=args.encoder_config,
            encoder_frozen=False,  # Unfrozen for fine-tuning
            conv1_channels=config['classifier_conv1_channels'],
            conv2_channels=config['classifier_conv2_channels'],
            kernel_size=config['classifier_kernel_size'],
            num_classes=config['num_classes'],
            dropout=config['classifier_dropout'],
            device=device
        )
        
    elif args.variant == 4:
        # Random + frozen encoder, train classifier only
        # Load encoder config for architecture
        with open(args.encoder_config, 'r') as f:
            encoder_config = json.load(f)
        encoder_config.pop('output_dim', None)
        
        model = create_classifier_from_scratch(
            encoder_config=encoder_config,
            encoder_frozen=True,  # Frozen random encoder
            conv1_channels=config['classifier_conv1_channels'],
            conv2_channels=config['classifier_conv2_channels'],
            kernel_size=config['classifier_kernel_size'],
            num_classes=config['num_classes'],
            dropout=config['classifier_dropout'],
            device=device
        )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier_head.parameters())
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Encoder parameters: {encoder_params:,}")
    logger.info(f"Classifier head parameters: {classifier_params:,}")
    logger.info(f"Encoder frozen: {model.encoder_frozen}")
    logger.info(f"Encoder output dim: {model.encoder.output_dim}")
    
    # Training for all variants
    trainer = ClassifierTrainer(
            model=model,
            optimizer_name=config['optimizer'],
            lr=config['lr'],
        device=device
    )
    
    # Training loop
    logger.info("Starting training...")
    metrics_history = []
    best_val_accuracy = 0.0
    best_val_accuracy_epoch = 0
    
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
        logger.info(
            f"  train_loss: {train_metrics['train_loss']:.6f} | "
            f"train_acc: {train_metrics['train_accuracy']:.2f}% | "
            f"val_loss: {val_metrics['val_loss']:.6f} | "
            f"val_acc: {val_metrics['val_accuracy']:.2f}%"
        )
        
        # Track best
        if val_metrics['val_accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['val_accuracy']
            best_val_accuracy_epoch = epoch + 1
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.out_path, 'best_model.pth'))
    
    # Final test evaluation
    logger.info("Running final test evaluation...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(args.out_path, 'metrics.csv'), index=False)
    logger.info(f"Metrics saved to {os.path.join(args.out_path, 'metrics.csv')}")
    
    # Save final model weights
    weights_path = os.path.join(args.out_path, 'classifier.pth')
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Classifier weights saved to {weights_path}")
    
    # Save model config
    model_config_path = os.path.join(args.out_path, 'model_config.json')
    with open(model_config_path, 'w') as f:
        json.dump(model.get_config(), f, indent=2)
    logger.info(f"Model config saved to {model_config_path}")
    
    # Save test results
    test_results = {
        'variant': args.variant,
        'variant_description': get_variant_description(args.variant),
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'final_train_loss': metrics_history[-1]['train_loss'],
        'final_train_accuracy': metrics_history[-1]['train_accuracy'],
        'final_val_loss': metrics_history[-1]['val_loss'],
        'final_val_accuracy': metrics_history[-1]['val_accuracy'],
        'best_val_accuracy': best_val_accuracy,
        'best_val_accuracy_epoch': best_val_accuracy_epoch,
    }
    
    with open(os.path.join(args.out_path, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Variant: {args.variant} - {get_variant_description(args.variant)}")
    logger.info(f"Final train loss: {metrics_history[-1]['train_loss']:.6f}")
    logger.info(f"Final train accuracy: {metrics_history[-1]['train_accuracy']:.2f}%")
    logger.info(f"Final val loss: {metrics_history[-1]['val_loss']:.6f}")
    logger.info(f"Final val accuracy: {metrics_history[-1]['val_accuracy']:.2f}%")
    logger.info(f"Best val accuracy: {best_val_accuracy:.2f}%")
    logger.info(f"Test loss: {test_metrics['loss']:.6f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info("=" * 60)
    
    # Print class names for reference
    logger.info("Class names:")
    for i, name in enumerate(CLASS_NAMES):
        logger.info(f"  {i}: {name}")


if __name__ == "__main__":
    main()
