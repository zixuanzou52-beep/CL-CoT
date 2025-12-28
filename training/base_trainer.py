"""Base trainer class for CL-CoT"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import ExperimentLogger, MetricsTracker
from utils.checkpoint import CheckpointManager
from utils.config import Config


class BaseTrainer:
    """Base trainer with common training utilities"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        logger: Optional[ExperimentLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: str = "cuda"
    ):
        """
        Initialize base trainer

        Args:
            model: Model to train
            config: Configuration object
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            logger: Experiment logger
            checkpoint_manager: Checkpoint manager
            device: Device to use
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = device

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')

        # Training settings
        self.max_grad_norm = config.get('training.max_grad_norm', 1.0)
        self.gradient_accumulation_steps = config.get(
            'training.gradient_accumulation_steps', 1
        )
        self.logging_steps = config.get('training.logging_steps', 100)
        self.eval_steps = config.get('training.eval_steps', 500)
        self.save_steps = config.get('training.save_steps', 1000)

        # Move model to device
        self.model.to(self.device)

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            Average epoch loss
        """
        raise NotImplementedError("Subclasses must implement train_epoch()")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def train(self, num_epochs: int):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.logger.info(f"{'='*50}")

            # Train
            train_loss = self.train_epoch()

            # Evaluate
            if self.eval_loader is not None:
                eval_metrics = self.evaluate()

                # Log metrics
                self.logger.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    **{f'eval_{k}': v for k, v in eval_metrics.items()}
                })

                # Save best model
                is_best = eval_metrics.get('loss', float('inf')) < self.best_metric
                if is_best:
                    self.best_metric = eval_metrics['loss']
                    self.logger.info(
                        f"New best model! Eval loss: {self.best_metric:.4f}"
                    )

                # Save checkpoint
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=epoch + 1,
                        metrics={'train_loss': train_loss, **eval_metrics},
                        is_best=is_best
                    )
            else:
                self.logger.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss
                })

        self.logger.info("Training complete!")

    def backward_step(self, loss: torch.Tensor):
        """
        Perform backward pass with gradient accumulation

        Args:
            loss: Loss tensor
        """
        # Scale loss by accumulation steps
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        # Update weights if accumulation steps reached
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Metrics dictionary
            step: Global step (if None, uses self.global_step)
        """
        if step is None:
            step = self.global_step

        # Update tracker
        self.metrics_tracker.update(metrics)

        # Log periodically
        if step % self.logging_steps == 0:
            avg_metrics = self.metrics_tracker.get_average(reset=True)

            log_str = f"Step {step}: " + ", ".join(
                [f"{k}: {v:.4f}" for k, v in avg_metrics.items()]
            )

            if self.scheduler:
                log_str += f", LR: {self.scheduler.get_last_lr()[0]:.6f}"

            self.logger.info(log_str)

            # Log to wandb
            self.logger.log({**avg_metrics, 'step': step})

    def save_model(self, save_path: str):
        """
        Save model

        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))
        self.logger.info(f"Model saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.global_step = checkpoint.get('step', 0)
            self.epoch = checkpoint.get('epoch', 0)

            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            raise ValueError("CheckpointManager not initialized")
