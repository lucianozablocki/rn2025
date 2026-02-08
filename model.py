"""
Convolutional Autoencoder Model

A configurable convolutional autoencoder for FashionMNIST.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with configurable architecture.
    
    Architecture:
        Encoder: N x [Conv2d + ReLU + MaxPool2d + Dropout] -> [Flatten + Linear + ReLU] (optional)
        Decoder: [Linear + ReLU + Unflatten] (optional) -> N x [ConvTranspose2d + ReLU/Sigmoid]
    """
    
    def __init__(
        self,
        num_layers=2,
        base_channels=16,
        kernel_size=3,
        latent_dim=64,
        dropout=0.2,
        use_linear=True,
        device='cpu'
    ):
        """
        Args:
            num_layers (int): Number of conv/deconv blocks (2 or 3).
            base_channels (int): Number of channels in first conv layer.
            kernel_size (int): Kernel size for conv layers (3 or 5).
            latent_dim (int): Size of the latent space (only used if use_linear=True).
            dropout (float): Dropout probability after each encoder conv block.
            use_linear (bool): Whether to use linear bottleneck layer.
            device (str): Device to run the model on.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_linear = use_linear
        self.device = device
        
        # Calculate padding to maintain spatial dimensions after conv
        padding = kernel_size // 2
        
        # Calculate spatial dimensions after each pooling layer
        # FashionMNIST: 28x28
        # After pool 1: 14x14
        # After pool 2: 7x7
        # After pool 3: 3x3 (7//2=3)
        spatial_sizes = [28, 14, 7, 3]
        self.final_spatial = spatial_sizes[num_layers]
        
        # Channel progression: 1 -> base -> base*2 -> base*4 ...
        channels = [1] + [base_channels * (2 ** i) for i in range(num_layers)]
        self.channels = channels
        self.final_channels = channels[-1]
        
        # Build encoder
        encoder_layers = []
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout)
            ])
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Flatten dimension after conv encoder
        self.flat_dim = self.final_channels * self.final_spatial * self.final_spatial
        
        # Optional linear bottleneck
        if use_linear:
            self.encoder_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flat_dim, latent_dim),
                nn.ReLU(inplace=True)
            )
            self.decoder_linear = nn.Sequential(
                nn.Linear(latent_dim, self.flat_dim),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (self.final_channels, self.final_spatial, self.final_spatial))
            )
        else:
            self.encoder_linear = nn.Identity()
            self.decoder_linear = nn.Identity()
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        reversed_channels = channels[::-1]  # e.g., [32, 16, 1] for 2 layers
        
        # Output sizes for ConvTranspose to reverse the pooling
        output_sizes = spatial_sizes[num_layers-1::-1]  # e.g., [7, 14, 28] for 2 layers
        
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            decoder_layers.append(
                nn.ConvTranspose2d(
                    reversed_channels[i], 
                    reversed_channels[i+1],
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=1 if output_sizes[i] % 2 == 0 else 0
                )
            )
            if is_last:
                # Last layer uses Sigmoid to output values in [0, 1] range
                # (we'll need to adjust normalization in data_loader)
                decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(nn.ReLU(inplace=True))
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
        self.to(device)
    
    def encode(self, x):
        """Encode input to latent representation."""
        x = self.encoder_conv(x)
        x = self.encoder_linear(x)
        return x
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        z = self.decoder_linear(z)
        z = self.decoder_conv(z)
        return z
    
    def forward(self, x):
        """Forward pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def get_config(self):
        """Return model configuration as dict."""
        return {
            'num_layers': self.num_layers,
            'base_channels': self.base_channels,
            'kernel_size': self.kernel_size,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'use_linear': self.use_linear
        }


class AutoencoderTrainer:
    """
    Trainer class for the Convolutional Autoencoder.
    """
    
    def __init__(self, model, optimizer_name='adam', lr=1e-3, device='cpu'):
        """
        Args:
            model: ConvAutoencoder instance.
            optimizer_name (str): 'adam' or 'sgd'.
            lr (float): Learning rate.
            device (str): Device to run training on.
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.optimizer_name = optimizer_name
        
        # Setup optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'adam' or 'sgd'.")
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def fit(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            
        Returns:
            dict: Training metrics {'loss': float}
        """
        self.model.train()
        total_loss = 0.0
        
        for images, _ in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            
            # Forward pass
            reconstructed = self.model(images)
            loss = self.criterion(reconstructed, images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return {'loss': avg_loss}
    
    def evaluate(self, data_loader):
        """
        Evaluate on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data.
            
        Returns:
            dict: Evaluation metrics {'loss': float}
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                reconstructed = self.model(images)
                loss = self.criterion(reconstructed, images)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return {'loss': avg_loss}


if __name__ == "__main__":
    # Test model instantiation
    print("Testing ConvAutoencoder...")
    
    # Test with 2 layers
    model = ConvAutoencoder(num_layers=2, base_channels=16, use_linear=True, latent_dim=64)
    print(f"\n2-layer model:")
    print(f"  Channels: {model.channels}")
    print(f"  Final spatial: {model.final_spatial}x{model.final_spatial}")
    print(f"  Flat dim: {model.flat_dim}")
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    # Test with 3 layers
    model3 = ConvAutoencoder(num_layers=3, base_channels=16, use_linear=True, latent_dim=32)
    print(f"\n3-layer model:")
    print(f"  Channels: {model3.channels}")
    print(f"  Final spatial: {model3.final_spatial}x{model3.final_spatial}")
    print(f"  Flat dim: {model3.flat_dim}")
    
    y3 = model3(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y3.shape}")
    
    # Test without linear layer
    model_no_linear = ConvAutoencoder(num_layers=2, use_linear=False)
    print(f"\n2-layer model (no linear):")
    y_no_lin = model_no_linear(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_no_lin.shape}")
    
    print("\nAll tests passed!")
