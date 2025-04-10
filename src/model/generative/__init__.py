"""
This module contains the implementation of the Generative network.
"""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100) -> None:
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
