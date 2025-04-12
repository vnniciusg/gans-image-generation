"""
This module provides the class config for the project configurations
"""


class Config:
    EPOCHS: int = 100
    NOISE_DIM: int = 100
    LEARNING_RATE: float = 2e-4
    BETA_1: float = 0.5
    SAVE_INTERVAL: int = 100
    SAVE_LOG_INTERVAL: int = 50
