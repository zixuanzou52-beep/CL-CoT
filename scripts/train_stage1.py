"""Stage 1: Supervised pre-training script"""
import argparse
import torch
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config
from utils.logger import ExperimentLogger, MetricsTracker
from utils.checkpoint import CheckpointManager
from models.base_model import CLCoTModel
from data.dataset_loader import load_dataset, collate_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stage 1: Supervised Training')

    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wtq', 'tabfact', 'hybridqa'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    logger,
    metrics_tracker,
    device,
    max_grad_norm
):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Create labels (shift input_ids by 1)
        labels = input_ids.clone()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track metrics
        metrics_tracker.update({'loss': loss.item()})
        epoch_loss += loss.item()

        # Log periodically
        if (batch_idx + 1) % 10 == 0:
            avg_metrics = metrics_tracker.get_average(reset=True)
            logger.info(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {avg_metrics['loss']:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs['loss'].item()

    return total_loss / len(dataloader)


def main():
    """Main training function"""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = Config(args.config)

    # Override config with command line args
    if args.num_epochs:
        config.set('training.stage1_epochs', args.num_epochs)
    if args.batch_size:
        config.set('training.stage1_batch_size', args.batch_size)
    if args.learning_rate:
        config.set('training.stage1_lr', args.learning_rate)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(str(output_dir / 'config.yaml'))

    # Initialize logger
    logger = ExperimentLogger(
        project_name=config.get('wandb.project', 'CL-CoT'),
        run_name=f"stage1_{args.dataset}_{args.seed}",
        config=config.to_dict(),
        log_dir=str(output_dir / 'logs'),
        use_wandb=config.get('wandb.enabled', False)
    )

    logger.info(f"Starting Stage 1 training on {args.dataset}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize model
    logger.info("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLCoTModel(
        model_name=config.get('model.base_model'),
        use_lora=config.get('model.use_lora'),
        lora_rank=config.get('model.lora_rank'),
        lora_alpha=config.get('model.lora_alpha'),
        lora_dropout=config.get('model.lora_dropout'),
        device=device
    )

    logger.info(f"Trainable parameters: {model.get_trainable_parameters():,}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(
        dataset_name=args.dataset,
        split='train',
        tokenizer=model.tokenizer,
        data_dir=args.data_dir,
        max_length=config.get('training.max_length')
    )

    eval_dataset = load_dataset(
        dataset_name=args.dataset,
        split='dev',
        tokenizer=model.tokenizer,
        data_dir=args.data_dir,
        max_length=config.get('training.max_length')
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training.stage1_batch_size'),
        shuffle=True,
        num_workers=config.get('data.num_workers', 0),
        collate_fn=collate_fn
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.get('training.stage1_batch_size'),
        shuffle=False,
        num_workers=config.get('data.num_workers', 0),
        collate_fn=collate_fn
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('training.stage1_lr'),
        weight_decay=config.get('training.weight_decay')
    )

    # Initialize scheduler
    num_epochs = config.get('training.stage1_epochs')
    num_training_steps = len(train_loader) * num_epochs
    warmup_steps = config.get('training.stage1_warmup_steps')

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir),
        max_keep=3,
        save_best=True
    )

    # Training loop
    logger.info("Starting training...")
    best_eval_loss = float('inf')
    metrics_tracker = MetricsTracker()

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            metrics_tracker=metrics_tracker,
            device=device,
            max_grad_norm=config.get('training.max_grad_norm')
        )

        # Evaluate
        eval_loss = evaluate(model, eval_loader, device)

        # Log metrics
        logger.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Eval Loss: {eval_loss:.4f}")

        # Save checkpoint
        is_best = eval_loss < best_eval_loss
        if is_best:
            best_eval_loss = eval_loss
            logger.info(f"New best model! Eval loss: {eval_loss:.4f}")

        checkpoint_manager.save_checkpoint(
            model=model.model,  # Save the actual model (not wrapper)
            optimizer=optimizer,
            scheduler=scheduler,
            step=epoch + 1,
            metrics={'train_loss': train_loss, 'eval_loss': eval_loss},
            is_best=is_best
        )

    # Save final model
    logger.info("Saving final model...")
    final_model_dir = output_dir / 'final'
    model.save_pretrained(str(final_model_dir))

    logger.info(f"Training complete! Best eval loss: {best_eval_loss:.4f}")
    logger.finish()


if __name__ == '__main__':
    main()
