"""
FashionMNIST Dataset Loader

This module provides utilities to load and prepare the FashionMNIST dataset
for training and testing a classifier.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# FashionMNIST class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_transforms():
    """
    Define the transformations for the FashionMNIST dataset.
    
    Returns:
        transform: A composition of transforms to apply to the images.
    """
    # Only convert to tensor [0, 1] - no normalization needed for autoencoder
    # since we use Sigmoid output in the decoder
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform


def load_fashion_mnist(data_dir="./data", batch_size=64, num_workers=2, val_split=0.2, seed=42):
    """
    Load the FashionMNIST dataset and create train/val/test data loaders.
    
    Args:
        data_dir (str): Directory to store/load the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        val_split (float): Fraction of training data to use for validation.
        seed (int): Random seed for reproducible train/val split.
    
    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
    """
    transform = get_transforms()
    
    # Download and load the full training data
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Split training data into train and validation sets
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Download and load the test data
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(train_loader, val_loader, test_loader):
    """
    Print information about the loaded datasets.
    
    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
    """
    print("=" * 50)
    print("FashionMNIST Dataset Information")
    print("=" * 50)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get a sample to show image dimensions
    images, labels = next(iter(train_loader))
    print(f"Image shape: {images.shape[1:]} (C, H, W)")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print("Loading FashionMNIST dataset...")
    
    train_loader, val_loader, test_loader = load_fashion_mnist(
        data_dir="./data",
        batch_size=64,
        num_workers=2,
        val_split=0.2
    )
    
    get_dataset_info(train_loader, val_loader, test_loader)
    
    # Verify data loading by getting a batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Images tensor shape: {images.shape}")
    print(f"  Labels tensor shape: {labels.shape}")
    print(f"  Sample labels: {[CLASS_NAMES[l] for l in labels[:5].tolist()]}")
