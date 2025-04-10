# GANs Image Generation

This project implements Generative Adversarial Networks (GANs) for image generation using PyTorch.

## Project Structure

```
├── main.py               # Entry point for the application
├── pyproject.toml        # Project configuration and dependencies
├── README.md             # Project documentation
├── src/                  # Source code for the project
│   ├── data/             # Data loading and preprocessing utilities
│   ├── model/            # GAN model components (Generator and Discriminator)
│   ├── train/            # Training loop and related utilities
│   └── utils/            # Helper functions and utilities
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.6.0
- TorchVision >= 0.21.0
- Matplotlib >= 3.10.1

## Usage

1. Install dependencies using `uv`:

   ```bash
   uv sync
   ```

2. Run the training script:

   ```bash
   uv run src/train/__init__.py
   ```

   Checkpoints will be saved in the `checkpoints` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [4 Best PyTorch Projects for Beginners](https://medium.com/@amit25173/4-best-pytorch-projects-for-beginners-b88049a44fa2)
