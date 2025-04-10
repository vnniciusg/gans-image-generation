"""
This module contains the data loading and preprocessing functions for the project.
"""

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose


def get_data_loader(batch_size: int = 64) -> DataLoader:
    """
    This function returns a DataLoader for the MNIST dataset.

    Args:
        batch_size (int): The number of samples per batch to load (default is 64).

    Returns:
        DataLoader: A DataLoader object for the MNIST dataset.
    """
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
