"""
Convolutional Autoencoder for FashionMNIST

Separate Encoder and Decoder classes for easy reuse.
The Encoder can be used as a feature extractor after training.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class Encoder(nn.Module):
    """
    Encoder network for the Convolutional Autoencoder.
    
    Can be used standalone as a feature extractor after training.
    
    Architecture:
        Conv2d + ReLU + MaxPool2d + Dropout
        [Optional: Conv2d + ReLU + Dropout]
        [Optional: Flatten + Linear + ReLU]
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
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_linear = use_linear
        self.use_extra_layer = use_extra_layer
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        
        # Calculate spatial size after first conv + pool
        # Conv2d with kernel k and no padding: 28 -> 28-k+1
        # MaxPool2d(2): (28-k+1) // 2
        self.spatial_after_conv1 = (28 - kernel_size + 1) // 2
        
        # Layer 1: Conv2d + ReLU + MaxPool2d + Dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout)
        )
        
        # Optional extra layer
        if use_extra_layer:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
            self.flat_dim = conv2_channels * self.spatial_after_conv1 * self.spatial_after_conv1
            self.out_channels = conv2_channels
        else:
            self.conv2 = nn.Identity()
            self.flat_dim = conv1_channels * self.spatial_after_conv1 * self.spatial_after_conv1
            self.out_channels = conv1_channels
        
        # Optional linear bottleneck (encoding part only)
        if use_linear:
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flat_dim, latent_dim),
                nn.ReLU()
            )
            self.output_dim = latent_dim
        else:
            self.linear = nn.Flatten()
            self.output_dim = self.flat_dim
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x)
        return x
    
    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'use_linear': self.use_linear,
            'use_extra_layer': self.use_extra_layer,
            'conv1_channels': self.conv1_channels,
            'conv2_channels': self.conv2_channels,
            'kernel_size': self.kernel_size,
            'output_dim': self.output_dim
        }


class Decoder(nn.Module):
    """
    Decoder network for the Convolutional Autoencoder.
    
    Architecture:
        [Optional: Linear + ReLU]
        Unflatten
        [Optional: ConvTranspose2d + ReLU]
        ConvTranspose2d + Sigmoid
    """
    
    def __init__(
        self,
        latent_dim=128,
        use_linear=True,
        use_extra_layer=False,
        conv1_channels=16,
        conv2_channels=8,
        kernel_size=3,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_linear = use_linear
        self.use_extra_layer = use_extra_layer
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        
        # Calculate dimensions
        self.spatial_after_conv1 = (28 - kernel_size + 1) // 2
        
        if use_extra_layer:
            self.flat_dim = conv2_channels * self.spatial_after_conv1 * self.spatial_after_conv1
            self.unflatten_channels = conv2_channels
        else:
            self.flat_dim = conv1_channels * self.spatial_after_conv1 * self.spatial_after_conv1
            self.unflatten_channels = conv1_channels
        
        # Optional linear layer (decoding part)
        if use_linear:
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, self.flat_dim),
                nn.ReLU()
            )
        else:
            self.linear = nn.Identity()
        
        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.unflatten_channels, self.spatial_after_conv1, self.spatial_after_conv1))
        
        # Optional extra layer
        if use_extra_layer:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=conv2_channels, out_channels=conv1_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.deconv1 = nn.Identity()
        
        # Final deconv layer
        deconv_kernel = 28 - (self.spatial_after_conv1 - 1) * 2 + 2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=conv1_channels, out_channels=1, kernel_size=deconv_kernel, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        z = self.linear(z)
        z = self.unflatten(z)
        z = self.deconv1(z)
        z = self.deconv2(z)
        return z


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder composed of separate Encoder and Decoder.
    
    After training, use `autoencoder.encoder` as a feature extractor.
    
    Example:
        # Train autoencoder
        autoencoder = ConvAutoencoder(latent_dim=128)
        # ... training ...
        
        # Use encoder for feature extraction
        features = autoencoder.encoder(images)
        
        # Or save/load encoder separately
        torch.save(autoencoder.encoder.state_dict(), 'encoder.pth')
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
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_linear = use_linear
        self.use_extra_layer = use_extra_layer
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        self.device = device
        
        # Create encoder and decoder
        self.encoder = Encoder(
            latent_dim=latent_dim,
            dropout=dropout,
            use_linear=use_linear,
            use_extra_layer=use_extra_layer,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            kernel_size=kernel_size
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            use_linear=use_linear,
            use_extra_layer=use_extra_layer,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            kernel_size=kernel_size
        )
        
        self.to(device)
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
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
        self.model = model
        self.device = device
        self.lr = lr
        self.optimizer_name = optimizer_name
        
        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'adam' or 'sgd'.")
        
        self.criterion = nn.MSELoss()
    
    def fit(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for images, _ in tqdm(train_loader):
            images = images.to(self.device)
            
            reconstructed = self.model(images)
            loss = self.criterion(reconstructed, images)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {'loss': total_loss / len(train_loader)}
    
    def evaluate(self, data_loader):
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                reconstructed = self.model(images)
                loss = self.criterion(reconstructed, images)
                total_loss += loss.item()
        
        return {'loss': total_loss / len(data_loader)}


if __name__ == "__main__":
    print("Testing Autoencoder components...")
    
    x = torch.randn(4, 1, 28, 28)
    
    # Test encoder standalone
    print("\n=== Encoder ===")
    encoder = Encoder(latent_dim=128, use_linear=True)
    z = encoder(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {z.shape}")
    print(f"  Config: {encoder.get_config()}")
    
    # Test decoder standalone
    print("\n=== Decoder ===")
    decoder = Decoder(latent_dim=128, use_linear=True)
    x_recon = decoder(z)
    print(f"  Input: {z.shape}")
    print(f"  Output: {x_recon.shape}")
    
    # Test full autoencoder
    print("\n=== ConvAutoencoder ===")
    autoencoder = ConvAutoencoder(latent_dim=128, use_extra_layer=True)
    y = autoencoder(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Config: {autoencoder.get_config()}")
    
    # Show that encoder is accessible
    print("\n=== Using encoder for feature extraction ===")
    features = autoencoder.encoder(x)
    print(f"  Features shape: {features.shape}")
    print(f"  Encoder output dim: {autoencoder.encoder.output_dim}")
    
    # Parameter counts
    print("\n=== Parameter counts ===")
    print(f"  Encoder: {sum(p.numel() for p in autoencoder.encoder.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in autoencoder.decoder.parameters()):,}")
    print(f"  Total: {sum(p.numel() for p in autoencoder.parameters()):,}")
    
    print("\nAll tests passed!")
