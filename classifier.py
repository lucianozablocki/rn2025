"""
Classifier for FashionMNIST using Encoder as feature extractor.

Assumes encoder uses use_linear=True, outputting a flat latent_dim vector.
The classifier reshapes to spatial and applies 2 conv layers.

Supports 4 training variants:
1) encoder_pretrained=True, encoder_frozen=True   - Use pretrained encoder as fixed feature extractor
2) encoder_pretrained=False, encoder_frozen=False - Train everything from scratch
3) encoder_pretrained=True, encoder_frozen=False  - Fine-tune pretrained encoder with classifier
4) encoder_pretrained=False, encoder_frozen=True  - Baseline (random frozen encoder, no training)
"""

import math
import torch
import torch.nn as nn
from tqdm import tqdm

from autoencoder import Encoder


class ConvClassifierHead(nn.Module):
    """
    Convolutional classifier head for flat latent features.
    
    Takes latent_dim vector from encoder, reshapes to spatial,
    applies 2 conv layers, then outputs class logits.
    
    Architecture:
        Reshape (latent_dim -> C x H x W)
        -> Conv2d + BatchNorm + ReLU + Dropout
        -> Conv2d + BatchNorm + ReLU + Dropout
        -> AdaptiveAvgPool -> Flatten -> Linear (logits)
    """
    
    def __init__(
        self,
        latent_dim,
        conv1_channels=32,
        conv2_channels=64,
        kernel_size=3,
        num_classes=10,
        dropout=0.3,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Fixed spatial reshape for latent_dim=256: 16 channels × 4 × 4
        self.spatial_size = 4
        self.input_channels = latent_dim // (self.spatial_size * self.spatial_size)
        
        # Unflatten to spatial representation
        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(self.input_channels, self.spatial_size, self.spatial_size)
        )
        
        # Conv layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, conv1_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        
        # Conv layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        
        # Global average pooling + output
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv2_channels, num_classes)  # Raw logits
        )
    
    def forward(self, x):
        x = self.unflatten(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        return x
    
    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'spatial_size': self.spatial_size,
            'input_channels': self.input_channels,
            'conv1_channels': self.conv1_channels,
            'conv2_channels': self.conv2_channels,
            'kernel_size': self.kernel_size,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
        }


class FashionMNISTClassifier(nn.Module):
    """
    Full classifier model: Encoder (feature extractor) + ConvClassifierHead.
    
    Assumes encoder outputs flat latent_dim vector (use_linear=True).
    
    The encoder can be:
    - Pretrained and frozen (variant 1)
    - Random and trainable (variant 2)
    - Pretrained and trainable/fine-tuned (variant 3)
    - Random and frozen (variant 4)
    """
    
    def __init__(
        self,
        encoder: Encoder,
        encoder_frozen: bool = True,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        kernel_size: int = 3,
        num_classes: int = 10,
        dropout: float = 0.3,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.encoder = encoder
        self.encoder_frozen = encoder_frozen
        self.device = device
        
        # Freeze encoder if requested
        if encoder_frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Create classifier head (input is encoder's latent_dim)
        self.classifier_head = ConvClassifierHead(
            latent_dim=encoder.output_dim,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            kernel_size=kernel_size,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.to(device)
    
    def forward(self, x):
        # Extract features with encoder
        if self.encoder_frozen:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        # Classify
        logits = self.classifier_head(features)
        return logits
    
    def freeze_encoder(self):
        """Freeze encoder weights."""
        self.encoder_frozen = True
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for fine-tuning."""
        self.encoder_frozen = False
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_config(self):
        return {
            'encoder_config': self.encoder.get_config(),
            'encoder_frozen': self.encoder_frozen,
            'classifier_head_config': self.classifier_head.get_config(),
        }


class ClassifierTrainer:
    """
    Trainer class for the FashionMNIST Classifier.
    
    Uses CrossEntropyLoss (which includes LogSoftmax internally).
    Computes accuracy as the primary metric.
    """
    
    def __init__(self, model, optimizer_name='adam', lr=1e-3, device='cpu'):
        self.model = model
        self.device = device
        self.lr = lr
        self.optimizer_name = optimizer_name
        
        # Only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'adam' or 'sgd'.")
        
        # CrossEntropyLoss combines LogSoftmax + NLLLoss
        self.criterion = nn.CrossEntropyLoss()
    
    def fit(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        # Set encoder to eval mode if frozen (for BatchNorm/Dropout behavior)
        if self.model.encoder_frozen:
            self.model.encoder.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return {'loss': total_loss / len(train_loader), 'accuracy': accuracy}
    
    def evaluate(self, data_loader):
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return {'loss': total_loss / len(data_loader), 'accuracy': accuracy}


def create_classifier_from_pretrained(
    encoder_weights_path: str,
    encoder_config_path: str,
    encoder_frozen: bool = True,
    conv1_channels: int = 32,
    conv2_channels: int = 64,
    kernel_size: int = 3,
    num_classes: int = 10,
    dropout: float = 0.3,
    device: str = 'cpu'
):
    """
    Create a classifier with a pretrained encoder.
    
    Args:
        encoder_weights_path: Path to encoder weights (.pth)
        encoder_config_path: Path to encoder config (.json)
        encoder_frozen: Whether to freeze encoder weights
        conv1_channels: Channels in first conv layer
        conv2_channels: Channels in second conv layer
        kernel_size: Kernel size for conv layers
        num_classes: Number of output classes
        dropout: Dropout rate
        device: Device to use
    
    Returns:
        FashionMNISTClassifier instance
    """
    import json
    
    # Load encoder config
    with open(encoder_config_path, 'r') as f:
        encoder_config = json.load(f)
    
    # Remove output_dim if present (it's computed, not a constructor arg)
    encoder_config.pop('output_dim', None)
    
    # Create encoder
    encoder = Encoder(**encoder_config)
    
    # Load pretrained weights
    encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    
    # Create classifier
    classifier = FashionMNISTClassifier(
        encoder=encoder,
        encoder_frozen=encoder_frozen,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        kernel_size=kernel_size,
        num_classes=num_classes,
        dropout=dropout,
        device=device
    )
    
    return classifier


def create_classifier_from_scratch(
    encoder_config: dict,
    encoder_frozen: bool = False,
    conv1_channels: int = 32,
    conv2_channels: int = 64,
    kernel_size: int = 3,
    num_classes: int = 10,
    dropout: float = 0.3,
    device: str = 'cpu'
):
    """
    Create a classifier with a fresh (random) encoder.
    
    Args:
        encoder_config: Dict with encoder hyperparameters
        encoder_frozen: Whether to freeze encoder
        conv1_channels: Channels in first conv layer
        conv2_channels: Channels in second conv layer
        kernel_size: Kernel size for conv layers
        num_classes: Number of output classes
        dropout: Dropout rate
        device: Device to use
    
    Returns:
        FashionMNISTClassifier instance
    """
    # Remove output_dim if present
    encoder_config = encoder_config.copy()
    encoder_config.pop('output_dim', None)
    
    # Create fresh encoder
    encoder = Encoder(**encoder_config)
    
    # Create classifier
    classifier = FashionMNISTClassifier(
        encoder=encoder,
        encoder_frozen=encoder_frozen,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        kernel_size=kernel_size,
        num_classes=num_classes,
        dropout=dropout,
        device=device
    )
    
    return classifier


if __name__ == "__main__":
    print("Testing Classifier components...")
    
    x = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    
    # Test with fresh encoder (variant 2: from scratch)
    print("\n=== Variant 2: Encoder random + unfrozen ===")
    encoder = Encoder(latent_dim=128, use_linear=True)
    classifier = FashionMNISTClassifier(
        encoder=encoder,
        encoder_frozen=False,
        conv1_channels=32,
        conv2_channels=64,
    )
    logits = classifier(x)
    print(f"  Input: {x.shape}")
    print(f"  Encoder output dim: {encoder.output_dim}")
    print(f"  Classifier spatial reshape: {classifier.classifier_head.input_channels}x{classifier.classifier_head.spatial_size}x{classifier.classifier_head.spatial_size}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Encoder frozen: {classifier.encoder_frozen}")
    print(f"  Trainable params: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")
    
    # Test with frozen encoder (variant 1)
    print("\n=== Variant 1: Encoder pretrained + frozen ===")
    encoder2 = Encoder(latent_dim=128, use_linear=True)
    classifier2 = FashionMNISTClassifier(
        encoder=encoder2,
        encoder_frozen=True,
        conv1_channels=32,
        conv2_channels=64,
    )
    logits2 = classifier2(x)
    print(f"  Input: {x.shape}")
    print(f"  Output logits: {logits2.shape}")
    print(f"  Encoder frozen: {classifier2.encoder_frozen}")
    print(f"  Trainable params: {sum(p.numel() for p in classifier2.parameters() if p.requires_grad):,}")
    
    # Test loss computation
    print("\n=== Loss computation (CrossEntropyLoss) ===")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test accuracy computation
    print("\n=== Accuracy computation ===")
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == labels).float().mean() * 100
    print(f"  Accuracy: {accuracy:.2f}%")
    
    print("\n=== Config ===")
    print(f"  {classifier.get_config()}")
    
    print("\nAll tests passed!")
