"""Stage 3: Reinforcement learning fine-tuning script"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import load_dataset, collate_fn
from models.base_model import CLCoTModel
from training.stage3_trainer import Stage3Trainer
from utils.checkpoint import CheckpointManager
from utils.config import Config
from utils.logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: RL Fine-tuning (PPO)")

    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wtq", "tabfact", "hybridqa"],
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def load_model(config: Config, device: str, pretrained_model: str):
    pretrained_path = Path(pretrained_model)
    adapter_config = pretrained_path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(pretrained_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config.get("model.base_model"),
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(pretrained_path))
        wrapper = CLCoTModel(
            model_name=config.get("model.base_model"),
            use_lora=False,
            device=device,
        )
        wrapper.tokenizer = tokenizer
        wrapper.base_model = base_model
        wrapper.model = peft_model
        return wrapper

    return CLCoTModel(model_name=str(pretrained_path), device=device)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = Config(args.config)

    if args.batch_size is not None:
        config.set("training.stage3_batch_size", args.batch_size)
    if args.learning_rate is not None:
        config.set("training.stage3_lr", args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(output_dir / "config.yaml"))

    logger = ExperimentLogger(
        project_name=config.get("wandb.project", "CL-CoT"),
        run_name=f"stage3_{args.dataset}_{args.seed}",
        config=config.to_dict(),
        log_dir=str(output_dir / "logs"),
        use_wandb=config.get("wandb.enabled", False),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config=config, device=device, pretrained_model=args.pretrained_model)

    train_dataset = load_dataset(
        dataset_name=args.dataset,
        split="train",
        tokenizer=model.tokenizer,
        data_dir=args.data_dir,
        max_length=config.get("training.max_length"),
    )
    eval_dataset = load_dataset(
        dataset_name=args.dataset,
        split="dev",
        tokenizer=model.tokenizer,
        data_dir=args.data_dir,
        max_length=config.get("training.max_length"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("training.stage3_batch_size"),
        shuffle=True,
        num_workers=config.get("data.num_workers", 0),
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.get("training.stage3_batch_size"),
        shuffle=False,
        num_workers=config.get("data.num_workers", 0),
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=config.get("training.stage3_lr"),
        weight_decay=config.get("training.weight_decay", 0.01),
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        max_keep=3,
        save_best=True,
    )

    trainer = Stage3Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=None,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=device,
    )

    trainer.train(num_epochs=args.num_epochs)

    trainer.save_model(str(output_dir / "final"))
    torch.save(trainer.value_head.state_dict(), str(output_dir / "value_head.pt"))
    logger.finish()


if __name__ == "__main__":
    main()

