"""Checkpoint management utilities"""
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


class CheckpointManager:
    """Manage model checkpoints"""

    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 5,
        save_best: bool = True
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_keep = max_keep
        self.save_best = save_best
        self.best_metric = None
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            step: Current step
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append((step, checkpoint_path))

        # Save metrics
        metrics_path = self.checkpoint_dir / f"checkpoint-{step}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Manage checkpoint count
        if len(self.checkpoints) > self.max_keep:
            old_step, old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
            old_metrics = self.checkpoint_dir / f"checkpoint-{old_step}_metrics.json"
            if old_metrics.exists():
                old_metrics.unlink()

        # Save best checkpoint
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best"
            best_path.mkdir(exist_ok=True)
            best_checkpoint = best_path / "checkpoint.pt"
            shutil.copy(checkpoint_path, best_checkpoint)

            best_metrics = best_path / "metrics.json"
            with open(best_metrics, 'w') as f:
                json.dump(metrics, f, indent=2)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load checkpoint

        Args:
            model: Model to load into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Specific checkpoint path to load
            load_best: Whether to load best checkpoint

        Returns:
            Dictionary with step and metrics
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best" / "checkpoint.pt"
        elif checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoints:
                raise ValueError("No checkpoints found")
            _, checkpoint_path = self.checkpoints[-1]

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return {
            'step': checkpoint['step'],
            'metrics': checkpoint.get('metrics', {})
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1][1]

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        best_path = self.checkpoint_dir / "best" / "checkpoint.pt"
        if best_path.exists():
            return best_path
        return None

    def list_checkpoints(self) -> list:
        """List all checkpoints"""
        return [(step, str(path)) for step, path in self.checkpoints]


def save_model(model, save_dir: str):
    """
    Save model using HuggingFace format

    Args:
        model: Model to save
        save_dir: Directory to save to
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_dir)
    else:
        torch.save(model.state_dict(), save_path / "pytorch_model.bin")


def load_model(model, load_dir: str):
    """
    Load model from directory

    Args:
        model: Model to load into
        load_dir: Directory to load from
    """
    load_path = Path(load_dir)

    if hasattr(model, 'from_pretrained'):
        return model.from_pretrained(load_dir)
    else:
        model_file = load_path / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
        return model
