import torch
import typer
from pathlib import Path

from src.config import Config
from src.model.generative import Generator
from src.utils import generate_and_show_images


app = typer.Typer()


def load_model(checkpoint_path: Path, config: Config, device: torch.device) -> Generator:
    """
    Load the saved model for image generation.

    Args:
        checkpoint_path (Path): the path of the checkpoint file .pth

    Returns:
        Generator: Generator model loaded and ready for inference
    """
    generator = Generator(config.NOISE_DIM).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    return generator


@app.command()
def main(checkpoint_path: Path = typer.Argument(..., help="path to the .pth checkpoint file")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    generator = load_model(checkpoint_path, config, device)
    generate_and_show_images(generator, device=device, noise_dim=config.NOISE_DIM)


if __name__ == "__main__":
    app()
