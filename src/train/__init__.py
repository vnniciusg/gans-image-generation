"""
This module contains the training loop for the GANs (Generative Adversarial Networks) model.
"""

import sys, pathlib, os
from pathlib import Path
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from src.data import get_data_loader
from src.config import Config
from src.model.generative import Generator
from src.model.discriminator import Discriminator


def train(data_loader: DataLoader, save_dir: Path = Path("checkpoints"), log_dir: Path = Path("logs")) -> None:
    """
    Trains the GAN model using the provided data loader.

    Args:
        data_loader (DataLoader): DataLoader for the training dataset.
        save_dir (Path): Directory to save the model checkpoints.
        log_dir (Path): Directory to save the TensorBoard logs.
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = save_dir / timestamp
    log_dir = log_dir / timestamp

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator(config.NOISE_DIM).to(device)
    discriminator = Discriminator().to(device)

    d_optim = Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA_1, 0.999))
    g_optim = Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA_1, 0.999))

    criterion = nn.BCELoss()
    writer = SummaryWriter(log_dir)

    best_g_loss = float("inf")

    for epoch in range(config.EPOCHS):
        d_losses = []
        g_losses = []
        real_accs = []
        fake_accs = []

        for batch_idx, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(device)

            # --- Train Discriminator ---
            d_optim.zero_grad()

            real_pred = discriminator(real_images)
            real_loss = criterion(real_pred, torch.ones(batch_size, 1).to(device))
            real_acc = (real_pred >= 0.5).float().mean()

            noise = torch.randn(batch_size, config.NOISE_DIM)
            fake_images = generator(noise)
            fake_pred = discriminator(fake_images.detach())
            fake_loss = criterion(fake_pred, torch.zeros(batch_size, 1).to(device))
            fake_acc = (fake_pred < 0.5).float().mean()

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optim.step()

            # --- Train Generator ---
            g_optim.zero_grad()

            gen_pred = discriminator(fake_images)
            g_loss = criterion(gen_pred, torch.ones(batch_size, 1).to(device))
            g_loss.backward()
            g_optim.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_accs.append(real_acc.item())
            fake_accs.append(fake_acc.item())

            global_step = epoch * len(data_loader) + batch_idx

            if batch_idx % config.SAVE_LOG_INTERVAL == 0:
                writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
                writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
                writer.add_scalar("Accuracy/Real", real_acc.item(), global_step)
                writer.add_scalar("Accuracy/Fake", fake_acc.item(), global_step)

                logger.info(
                    f"Epoch [{epoch + 1}/{config.EPOCHS}] Batch [{batch_idx}/{len(data_loader)}] "
                    f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f} "
                    f"Real Acc: {real_acc.item():.2f} Fake Acc: {fake_acc.item():.2f}"
                )

            if global_step % config.SAVE_INTERVAL == 0:
                checkpoint = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "g_loss": g_loss.item(),
                    "d_loss": d_loss.item(),
                }
                torch.save(checkpoint, f"{save_dir}/checkpoint_{timestamp}_step{global_step}.pth")

        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_real_acc = sum(real_accs) / len(real_accs)
        avg_fake_acc = sum(fake_accs) / len(fake_accs)

        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "epoch": epoch,
                    "g_loss": avg_g_loss,
                    "d_loss": avg_d_loss,
                },
                f"{save_dir}/best_model_{timestamp}.pth",
            )

        writer.add_scalar("Epoch Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Epoch Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Epoch Accuracy/Real", avg_real_acc, epoch)
        writer.add_scalar("Epoch Accuracy/Fake", avg_fake_acc, epoch)

        logger.info(
            f"End of Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Avg D Loss: {avg_d_loss:.4f} Avg G Loss: {avg_g_loss:.4f} "
            f"Avg Real Acc: {avg_real_acc:.2f} Avg Fake Acc: {avg_fake_acc:.2f}"
        )

        torch.save(
            {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "epoch": config.EPOCHS},
            f"{save_dir}/final_model_{timestamp}.pth",
        )

        writer.close()


if __name__ == "__main__":
    train(get_data_loader())
