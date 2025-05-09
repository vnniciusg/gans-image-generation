"""
This module contains utility functions for generating and displaying images using a generator model.
"""

import torch
import torchvision
from matplotlib import pyplot as plt

from src.model.generative import Generator


def generate_and_show_images(
    generator: Generator, device: torch.device, num_images: int = 10, noise_dim: int = 100
) -> None:
    """
    This function generates images using the provided generator model and displays them.

    Args:
        generator (Generator): The generator model used to create images.
        num_images (int): The number of images to generate.
        noise_dim (int): The dimension of the noise vector used for generation.
    """
    generator.to(device)
    noise = torch.rand(num_images, noise_dim).to(device)
    generated_images = generator(noise)

    generated_images = generated_images.view(num_images, 1, 28, 28).to(device)
    grid = torchvision.utils.make_grid(generated_images, nrow=5, normalize=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
    plt.show()
