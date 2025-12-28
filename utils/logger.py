"""Logging utilities for CL-CoT"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """
    Experiment logger with Weights & Biases integration
    """

    def __init__(
        self,
        project_name: str,
        run_name: str,
        config: Dict[str, Any],
        log_dir: str = "logs",
        use_wandb: bool = True,
        wandb_entity: Optional[str] = None
    ):
        """
        Initialize experiment logger

        Args:
            project_name: W&B project name
            run_name: W&B run name
            config: Configuration dictionary
            log_dir: Local log directory
            use_wandb: Whether to use W&B
            wandb_entity: W&B entity name
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup local logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{run_name}_{timestamp}.log"
        self.logger = setup_logger(run_name, str(log_file))

        # Initialize W&B
        if self.use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    entity=wandb_entity
                )
                self.logger.info("W&B initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        # Log to W&B
        if self.use_wandb:
            try:
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as e:
                self.logger.warning(f"Failed to log to W&B: {e}")

        # Log to console/file
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items()])
        if step is not None:
            self.logger.info(f"Step {step}: {metric_str}")
        else:
            self.logger.info(f"Metrics: {metric_str}")

    def log_table(self, name: str, data: Any):
        """
        Log table data

        Args:
            name: Table name
            data: Table data (DataFrame or list of lists)
        """
        if self.use_wandb:
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    table = wandb.Table(dataframe=data)
                else:
                    table = wandb.Table(data=data)
                wandb.log({name: table})
            except Exception as e:
                self.logger.warning(f"Failed to log table to W&B: {e}")

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """
        Log artifact to W&B

        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
        """
        if self.use_wandb:
            try:
                artifact = wandb.Artifact(
                    name=f"{self.run_name}_{artifact_type}",
                    type=artifact_type
                )
                artifact.add_file(artifact_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to W&B: {e}")

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def finish(self):
        """Finish logging and cleanup"""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish W&B: {e}")


class MetricsTracker:
    """Track and aggregate metrics over training"""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        """
        Update metrics

        Args:
            metrics: Dictionary of metrics to update
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def get_average(self, reset: bool = True) -> Dict[str, float]:
        """
        Get average of accumulated metrics

        Args:
            reset: Whether to reset metrics after getting average

        Returns:
            Dictionary of averaged metrics
        """
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
            else:
                averages[key] = 0.0

        if reset:
            self.reset()

        return averages

    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}
