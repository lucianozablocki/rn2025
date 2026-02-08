"""
Convolutional Autoencoder Model

A configurable convolutional autoencoder for FashionMNIST.
Based on the course assignment specification.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for FashionMNIST.
    
    Base architecture:
        Encoder Conv: 1ch 28x28 -> Conv2d(kernel) -> Cch -> ReLU -> MaxPool2d(2x2) -> Cch 13x13 -> Dropout
        Bottleneck: Flatten -> Linear(->n) -> ReLU -> Linear(n->) -> ReLU
        Decoder: Unflatten -> ConvTranspose2d -> 1ch 28x28 -> Sigmoid
    
    With extra layer (use_extra_layer=True):
        Adds an additional conv/deconv layer between the first layer and bottleneck.
    """
    
    def __init__(
        self,
        latent_dim=128,
        dropout=0.2,
        use_linear=True,
        use_extra_layer=False,
        conv1_channels=16,
        conv2_channels=8,
        kernel_size=3,
        device='cpu'
    ):
        """
        Args:
            latent_dim (int): Size of the latent space (bottleneck dimension n).
            dropout (float): Dropout probability.
            use_linear (bool): Whether to use linear bottleneck layer.
            use_extra_layer (bool): Whether to add extra conv/deconv layer.
            conv1_channels (int): Number of output channels for first conv layer (default 16).
            conv2_channels (int): Number of output channels for extra layer (default 8).
            kernel_size (int): Kernel size for conv layers (default 3).
            device (str): Device to run the model on.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_linear = use_linear
        self.use_extra_layer = use_extra_layer
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        self.device = device
        
        # Calculate spatial size after first conv + pool
        # Conv2d with kernel k and no padding: 28 -> 28-k+1
        # MaxPool2d(2): (28-k+1) // 2
        self.spatial_after_conv1 = (28 - kernel_size + 1) // 2  # e.g., kernel=3 -> 26//2=13
        
        # ============== ENCODER ==============
        # Layer 1: Conv2d + ReLU + MaxPool2d + Dropout
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout)
        )
        
        # Optional extra layer: Conv2d (keeps spatial size) + ReLU + Dropout
        if use_extra_layer:
            self.encoder_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
            self.flat_dim = conv2_channels * self.spatial_after_conv1 * self.spatial_after_conv1
            self.unflatten_channels = conv2_channels
        else:
            self.encoder_conv2 = nn.Identity()
            self.flat_dim = conv1_channels * self.spatial_after_conv1 * self.spatial_after_conv1
            self.unflatten_channels = conv1_channels
        
        # ============== BOTTLENECK ==============
        if use_linear:
            self.bottleneck = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flat_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, self.flat_dim),
                nn.ReLU()
            )
        else:
            self.bottleneck = nn.Flatten()
        
        # ============== DECODER ==============
        # Unflatten back to (channels, spatial, spatial)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.unflatten_channels, self.spatial_after_conv1, self.spatial_after_conv1))
        
        # Optional extra layer: ConvTranspose2d (conv2_channels -> conv1_channels) + ReLU
        if use_extra_layer:
            self.decoder_conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=conv2_channels, out_channels=conv1_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.decoder_conv1 = nn.Identity()
        
        # Final layer: ConvTranspose2d -> 1ch 28x28 -> Sigmoid
        # Need to upsample from spatial_after_conv1 back to 28
        # Using kernel=6, stride=2, padding=1 works for 13->28 (when kernel_size=3)
        # General formula: output = (input - 1) * stride - 2*padding + kernel
        # For 13->28: 28 = (13-1)*2 - 2*1 + k => k = 28 - 24 + 2 = 6
        # Calculate the right kernel for the deconv
        deconv_kernel = 28 - (self.spatial_after_conv1 - 1) * 2 + 2  # stride=2, padding=1
        
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=conv1_channels, out_channels=1, kernel_size=deconv_kernel, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def encode(self, x):
        """Encode input to latent/flattened representation."""
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        x = self.bottleneck(x)
        return x
    
    def decode(self, z):
        """Decode from latent/flattened representation."""
        z = self.unflatten(z)
        z = self.decoder_conv1(z)
        z = self.decoder_conv2(z)
        return z
    
    def forward(self, x):
        """Forward pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def get_config(self):
        """Return model configuration as dict."""
        return {
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'use_linear': self.use_linear,
            'use_extra_layer': self.use_extra_layer,
            'conv1_channels': self.conv1_channels,
            'conv2_channels': self.conv2_channels,
            'kernel_size': self.kernel_size
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
    
    x = torch.randn(4, 1, 28, 28)
    
    # Test base model (no extra layer)
    print("\n=== Base model (1 conv layer, kernel=3) ===")
    model_base = ConvAutoencoder(latent_dim=128, dropout=0.2, use_linear=True, use_extra_layer=False, conv1_channels=16, kernel_size=3)
    print(f"  Config: {model_base.get_config()}")
    print(f"  Spatial after conv1: {model_base.spatial_after_conv1}")
    print(f"  Flat dim: {model_base.flat_dim}")
    y_base = model_base(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_base.shape}")
    
    # Test with extra layer
    print("\n=== Model with extra layer (2 conv layers) ===")
    model_extra = ConvAutoencoder(latent_dim=128, dropout=0.2, use_linear=True, use_extra_layer=True, conv1_channels=16, conv2_channels=8)
    print(f"  Config: {model_extra.get_config()}")
    print(f"  Flat dim: {model_extra.flat_dim}")
    y_extra = model_extra(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_extra.shape}")
    
    # Test with different kernel size
    print("\n=== Model with kernel_size=5 ===")
    model_k5 = ConvAutoencoder(latent_dim=64, kernel_size=5, conv1_channels=32)
    print(f"  Config: {model_k5.get_config()}")
    print(f"  Spatial after conv1: {model_k5.spatial_after_conv1}")
    print(f"  Flat dim: {model_k5.flat_dim}")
    y_k5 = model_k5(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_k5.shape}")
    
    # Test without linear bottleneck
    print("\n=== Model without linear bottleneck ===")
    model_no_linear = ConvAutoencoder(use_linear=False, use_extra_layer=False)
    print(f"  Config: {model_no_linear.get_config()}")
    y_no_lin = model_no_linear(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_no_lin.shape}")
    
    # Count parameters
    print("\n=== Parameter counts ===")
    for name, m in [("Base (k=3, c=16)", model_base), ("Extra layer", model_extra), ("k=5, c=32", model_k5), ("No linear", model_no_linear)]:
        params = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {params:,} parameters")
    
    print("\nAll tests passed!")
