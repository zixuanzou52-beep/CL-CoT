"""Stage 2: Contrastive learning training script"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import ContrastiveDataset
from models.base_model import CLCoTModel
from training.stage2_trainer import Stage2Trainer
from utils.checkpoint import CheckpointManager
from utils.config import Config
from utils.logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Contrastive Learning")

    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wtq", "tabfact", "hybridqa"],
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
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


def resolve_data_path(data_dir: str, dataset: str, split: str, prefer_suffix: str) -> Path:
    base_dir = Path(data_dir) / dataset
    preferred = base_dir / f"{split}{prefer_suffix}.json"
    fallback = base_dir / f"{split}.json"
    if preferred.exists():
        return preferred
    return fallback


def contrastive_collate_fn(batch):
    return {
        "table": [item["table"] for item in batch],
        "question": [item["question"] for item in batch],
        "positive_path": [item["positive_path"] for item in batch],
        "negative_paths": [item["negative_paths"] for item in batch],
        "answer": [item.get("answer", "") for item in batch],
    }


def load_model(config: Config, device: str, pretrained_model: Optional[str]):
    if not pretrained_model:
        return CLCoTModel(
            model_name=config.get("model.base_model"),
            use_lora=config.get("model.use_lora"),
            lora_rank=config.get("model.lora_rank"),
            lora_alpha=config.get("model.lora_alpha"),
            lora_dropout=config.get("model.lora_dropout"),
            device=device,
        )

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

    if args.num_epochs is not None:
        config.set("training.stage2_epochs", args.num_epochs)
    if args.batch_size is not None:
        config.set("training.stage2_batch_size", args.batch_size)
    if args.learning_rate is not None:
        config.set("training.stage2_lr", args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(output_dir / "config.yaml"))

    logger = ExperimentLogger(
        project_name=config.get("wandb.project", "CL-CoT"),
        run_name=f"stage2_{args.dataset}_{args.seed}",
        config=config.to_dict(),
        log_dir=str(output_dir / "logs"),
        use_wandb=config.get("wandb.enabled", False),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config=config, device=device, pretrained_model=args.pretrained_model)

    if args.train_path:
        train_path = Path(args.train_path)
    else:
        train_path = resolve_data_path(args.data_dir, args.dataset, "train", "_with_negatives")

    if args.eval_path:
        eval_path = Path(args.eval_path)
    else:
        eval_candidate = resolve_data_path(args.data_dir, args.dataset, "dev", "_with_negatives")
        eval_path = eval_candidate if eval_candidate.exists() else None

    train_dataset = ContrastiveDataset(
        data_path=str(train_path),
        tokenizer=model.tokenizer,
        negative_ratio=config.get("contrastive.negative_ratio", 5),
    )
    eval_dataset = (
        ContrastiveDataset(
            data_path=str(eval_path),
            tokenizer=model.tokenizer,
            negative_ratio=config.get("contrastive.negative_ratio", 5),
        )
        if eval_path
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("training.stage2_batch_size"),
        shuffle=True,
        num_workers=config.get("data.num_workers", 0),
        collate_fn=contrastive_collate_fn,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=config.get("training.stage2_batch_size"),
            shuffle=False,
            num_workers=config.get("data.num_workers", 0),
            collate_fn=contrastive_collate_fn,
        )
        if eval_dataset
        else None
    )

    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=config.get("training.stage2_lr"),
        weight_decay=config.get("training.weight_decay", 0.01),
    )

    num_epochs = config.get("training.stage2_epochs")
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = config.get("training.stage2_warmup_steps")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        max_keep=3,
        save_best=True,
    )

    trainer = Stage2Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=device,
    )

    trainer.train(num_epochs=num_epochs)

    trainer.save_model(str(output_dir / "final"))
    torch.save(trainer.path_encoder.state_dict(), str(output_dir / "path_encoder.pt"))
    torch.save(trainer.contrastive_loss.state_dict(), str(output_dir / "contrastive_loss.pt"))
    logger.finish()


if __name__ == "__main__":
    main()

