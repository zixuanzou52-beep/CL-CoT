"""Stage 1: Supervised pre-training trainer"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_trainer import BaseTrainer
from utils.config import Config
from utils.logger import ExperimentLogger


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage 1: Supervised pre-training"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        logger: Optional[ExperimentLogger] = None,
        checkpoint_manager: Optional[Any] = None,
        device: str = "cuda"
    ):
        """
        Initialize Stage 1 trainer

        Args:
            model: CL-CoT model
            config: Configuration
            train_loader: Training dataloader
            eval_loader: Evaluation dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            logger: Logger
            checkpoint_manager: Checkpoint manager
            device: Device
        """
        super().__init__(
            model=model,
            config=config,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            device=device
        )

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            Average epoch loss
        """
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}",
            disable=not self.logger
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Create labels (for causal LM, labels = input_ids shifted)
            labels = input_ids.clone()

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']

            # Backward pass
            self.backward_step(loss)

            # Track metrics
            epoch_loss += loss.item()
            self.global_step += 1

            # Log metrics
            self.log_metrics({'loss': loss.item()})

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Periodic evaluation
            if self.eval_loader and self.global_step % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.logger.info(
                    f"Step {self.global_step} - Eval: {eval_metrics}"
                )
                self.model.train()

            # Periodic checkpoint
            if self.checkpoint_manager and self.global_step % self.save_steps == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    metrics={'train_loss': loss.item()},
                    is_best=False
                )

        return epoch_loss / num_batches

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model

        Returns:
            Evaluation metrics
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = input_ids.clone()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']
                batch_size = input_ids.size(0)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }

    def generate_sample(
        self,
        table: Any,
        question: str,
        max_length: int = 256
    ) -> str:
        """
        Generate answer for a sample

        Args:
            table: Table object
            question: Question string
            max_length: Maximum generation length

        Returns:
            Generated answer
        """
        self.model.eval()

        # Create input
        from data.table_parser import TableParser
        parser = TableParser()
        table_text = parser.linearize(table, format="markdown")
        input_text = f"Table:\n{table_text}\n\nQuestion: {question}\n\nAnswer:"

        # Tokenize
        inputs = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=self.config.get('inference.temperature', 0.7),
                top_p=self.config.get('inference.top_p', 0.9),
                do_sample=True
            )

        # Decode
        generated_text = self.model.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Extract answer (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text

        return answer
