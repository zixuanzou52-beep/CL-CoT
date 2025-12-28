"""Stage 2: Contrastive learning trainer"""
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
from models.path_encoder import HierarchicalPathEncoder, TextEncoder
from models.contrastive_loss import ContrastiveLoss
from models.similarity import MultiDimensionalSimilarity


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2: Contrastive learning"""

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
        Initialize Stage 2 trainer

        Args:
            model: CL-CoT model
            config: Configuration
            train_loader: Training dataloader (ContrastiveDataset)
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

        # Initialize path encoder
        self.path_encoder = HierarchicalPathEncoder(
            hidden_dim=config.get('encoder.hidden_dim', 768),
            num_layers=config.get('encoder.num_layers', 6),
            num_heads=config.get('encoder.num_heads', 12),
            dropout=config.get('encoder.dropout', 0.1)
        ).to(device)

        # Initialize text encoder (for embeddings)
        self.text_encoder = TextEncoder(
            tokenizer=model.tokenizer,
            model=model.base_model,
            hidden_dim=config.get('encoder.hidden_dim', 768)
        )

        # Initialize contrastive loss
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.get('contrastive.temperature', 0.07),
            memory_bank_size=config.get('contrastive.memory_bank_size', 10000),
            momentum=config.get('contrastive.momentum', 0.999)
        ).to(device)

        # Initialize similarity function
        self.similarity = MultiDimensionalSimilarity(
            struct_weight=config.get('similarity.struct_weight', 0.3),
            semantic_weight=config.get('similarity.semantic_weight', 0.5),
            op_weight=config.get('similarity.op_weight', 0.2)
        )

        # Add path encoder to optimizer
        if self.optimizer:
            self.optimizer.add_param_group({
                'params': self.path_encoder.parameters(),
                'lr': config.get('training.stage2_lr')
            })

    def encode_path(
        self,
        reasoning_path: Any,
        table: Any,
        question: str
    ) -> torch.Tensor:
        """
        Encode a reasoning path

        Args:
            reasoning_path: ReasoningPath object
            table: Table object
            question: Question string

        Returns:
            Path embedding
        """
        # Encode table
        from data.table_parser import TableParser
        parser = TableParser()
        table_text = parser.linearize(table, format="markdown")
        table_emb = self.text_encoder.encode(table_text, device=self.device)

        # Encode question
        question_emb = self.text_encoder.encode(question, device=self.device)

        # Encode steps
        step_embeddings = []
        for step in reasoning_path.steps:
            step_text = step.to_text()
            step_emb_tokens = self.text_encoder.encode(
                step_text,
                device=self.device
            ).unsqueeze(0)  # [1, d] -> treat as sequence of 1
            step_embeddings.append(step_emb_tokens)

        # Encode path
        path_output = self.path_encoder(
            reasoning_path=reasoning_path,
            table_emb=table_emb,
            question_emb=question_emb,
            step_embeddings=step_embeddings
        )

        return path_output['path_embedding']

    def train_epoch(self) -> float:
        """
        Train for one epoch with contrastive learning

        Returns:
            Average epoch loss
        """
        self.model.train()
        self.path_encoder.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1} - Contrastive",
            disable=not self.logger
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            tables = batch['table']
            questions = batch['question']
            positive_paths = batch['positive_path']
            negative_paths_list = batch['negative_paths']

            batch_size = len(questions)
            total_loss = 0

            for i in range(batch_size):
                # Encode question
                question_emb = self.text_encoder.encode(
                    questions[i],
                    device=self.device
                )

                # Encode positive path
                positive_emb = self.encode_path(
                    positive_paths[i],
                    tables[i],
                    questions[i]
                )

                # Encode negative paths
                negative_embs = []
                for neg_path in negative_paths_list[i]:
                    neg_emb = self.encode_path(
                        neg_path,
                        tables[i],
                        questions[i]
                    )
                    negative_embs.append(neg_emb)

                negative_embs = torch.stack(negative_embs)  # [N, d]

                # Compute contrastive loss
                loss = self.contrastive_loss(
                    query_emb=question_emb,
                    positive_emb=positive_emb,
                    negative_embs=negative_embs
                )

                total_loss += loss

            # Average loss over batch
            loss = total_loss / batch_size

            # Backward pass
            self.backward_step(loss)

            # Track metrics
            epoch_loss += loss.item()
            self.global_step += 1

            # Log metrics
            self.log_metrics({'contrastive_loss': loss.item()})

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Periodic evaluation
            if self.eval_loader and self.global_step % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.logger.info(
                    f"Step {self.global_step} - Eval: {eval_metrics}"
                )
                self.model.train()
                self.path_encoder.train()

        return epoch_loss / num_batches

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate contrastive learning

        Returns:
            Evaluation metrics
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()
        self.path_encoder.eval()

        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                tables = batch['table']
                questions = batch['question']
                positive_paths = batch['positive_path']
                negative_paths_list = batch['negative_paths']

                batch_size = len(questions)

                for i in range(batch_size):
                    # Encode question
                    question_emb = self.text_encoder.encode(
                        questions[i],
                        device=self.device
                    )

                    # Encode positive
                    positive_emb = self.encode_path(
                        positive_paths[i],
                        tables[i],
                        questions[i]
                    )

                    # Encode negatives
                    negative_embs = []
                    for neg_path in negative_paths_list[i]:
                        neg_emb = self.encode_path(
                            neg_path,
                            tables[i],
                            questions[i]
                        )
                        negative_embs.append(neg_emb)

                    negative_embs = torch.stack(negative_embs)

                    # Compute loss
                    loss = self.contrastive_loss(
                        query_emb=question_emb,
                        positive_emb=positive_emb,
                        negative_embs=negative_embs
                    )

                    total_loss += loss.item()

                    # Compute accuracy (is positive closest to query?)
                    pos_sim = torch.cosine_similarity(
                        question_emb.unsqueeze(0),
                        positive_emb.unsqueeze(0)
                    )
                    neg_sims = torch.cosine_similarity(
                        question_emb.unsqueeze(0).expand(negative_embs.size(0), -1),
                        negative_embs
                    )

                    is_correct = (pos_sim > neg_sims.max()).float()
                    total_accuracy += is_correct.item()

                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
