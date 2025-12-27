"""Configuration management for CL-CoT"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class Config:
    """Configuration manager for CL-CoT experiments"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)

        Args:
            key: Configuration key (e.g., 'model.base_model')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary

        Args:
            updates: Dictionary of updates
        """
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self._config = recursive_update(self._config, updates)

    def save(self, path: str):
        """
        Save configuration to file

        Args:
            path: Output path
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting"""
        self.set(key, value)

    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"


@dataclass
class ModelConfig:
    """Model configuration"""
    base_model: str = "meta-llama/Llama-2-13b-hf"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class EncoderConfig:
    """Encoder configuration"""
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    dropout: float = 0.1
    max_position_embeddings: int = 512


@dataclass
class TrainingConfig:
    """Training configuration"""
    stage1_lr: float = 2e-5
    stage1_epochs: int = 3
    stage1_batch_size: int = 32
    stage2_lr: float = 1e-5
    stage2_epochs: int = 5
    stage2_batch_size: int = 64
    stage3_lr: float = 3e-6
    stage3_batch_size: int = 16
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    fp16: bool = True
    max_length: int = 512


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file

    Args:
        config_path: Path to config file

    Returns:
        Config object
    """
    return Config(config_path)
