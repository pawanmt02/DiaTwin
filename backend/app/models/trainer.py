"""
DiaTwin Training Pipeline

End-to-end training loop with early stopping, gradient clipping,
learning rate scheduling, and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader

from backend.app.models.forecaster import DiaTwinForecaster, QuantileLoss, count_parameters
from backend.app.core.config import config as cfg


class Trainer:
    """Manages model training with best practices for time-series forecasting."""

    def __init__(
        self,
        model: DiaTwinForecaster,
        device: str = "cpu",
        learning_rate: float = None,
        weight_decay: float = None,
    ):
        self.model = model
        self.device = device

        lr = learning_rate or cfg.model.learning_rate
        wd = weight_decay or cfg.model.weight_decay

        self.criterion = QuantileLoss(quantiles=model.quantiles)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        os.makedirs(cfg.model.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for static, dynamic, target in train_loader:
            static = static.to(self.device)
            dynamic = dynamic.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(static, dynamic)
            loss = self.criterion(predictions, target)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.model.gradient_clip_norm
            )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for static, dynamic, target in val_loader:
            static = static.to(self.device)
            dynamic = dynamic.to(self.device)
            target = target.to(self.device)

            predictions = self.model(static, dynamic)
            loss = self.criterion(predictions, target)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        patience: int = None,
    ) -> Dict:
        """Full training loop with early stopping.

        Returns:
            Dictionary with training history and best metrics.
        """
        num_epochs = num_epochs or cfg.model.num_epochs
        patience = patience or cfg.model.early_stopping_patience

        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training on {self.device} for up to {num_epochs} epochs")
        print(f"Early stopping patience: {patience}")
        print("-" * 60)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(cfg.model.best_model_path)
                print(f"  [OK] New best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break

        print("-" * 60)
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": len(self.train_losses),
            "quantiles": self.model.quantiles,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from {path}")
