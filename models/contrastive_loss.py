"""Contrastive loss for path learning"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for reasoning path quality learning

    L_contrast = -log[exp(Sim(h_Q, h_P+)/τ) / Σ exp(Sim(h_Q, h_P-)/τ)]
    """

    def __init__(
        self,
        temperature: float = 0.07,
        memory_bank_size: int = 10000,
        momentum: float = 0.999
    ):
        """
        Initialize contrastive loss

        Args:
            temperature: Temperature parameter for softmax
            memory_bank_size: Size of memory bank for negatives
            momentum: Momentum for memory bank updates
        """
        super().__init__()

        self.temperature = temperature
        self.memory_bank_size = memory_bank_size
        self.momentum = momentum

        # Memory bank for storing path embeddings
        self.register_buffer(
            'memory_bank',
            torch.randn(memory_bank_size, 768)
        )
        self.register_buffer(
            'memory_labels',
            torch.zeros(memory_bank_size, dtype=torch.long)
        )
        self.register_buffer(
            'memory_ptr',
            torch.zeros(1, dtype=torch.long)
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            query_emb: Query embedding (question) [d] or [B, d]
            positive_emb: Positive path embedding [d] or [B, d]
            negative_embs: Negative path embeddings [N, d] or [B, N, d]

        Returns:
            Scalar loss value
        """
        # Handle single sample vs batch
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if positive_emb.dim() == 1:
            positive_emb = positive_emb.unsqueeze(0)
        if negative_embs.dim() == 2:
            negative_embs = negative_embs.unsqueeze(0)

        batch_size = query_emb.size(0)

        # Normalize embeddings
        query_emb = F.normalize(query_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)
        negative_embs = F.normalize(negative_embs, dim=-1)

        # Compute positive similarity [B]
        pos_sim = torch.sum(query_emb * positive_emb, dim=-1) / self.temperature

        # Compute negative similarity [B, N]
        neg_sim = torch.bmm(
            negative_embs,
            query_emb.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # Concatenate positive and negative logits [B, N+1]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # Labels (positive is always first)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def update_memory_bank(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Update memory bank with new embeddings

        Args:
            embeddings: New embeddings [B, d]
            labels: Corresponding labels [B]
        """
        batch_size = embeddings.size(0)
        ptr = int(self.memory_ptr)

        # Circular buffer
        if ptr + batch_size <= self.memory_bank_size:
            self.memory_bank[ptr:ptr+batch_size] = embeddings.detach()
            self.memory_labels[ptr:ptr+batch_size] = labels
        else:
            # Wrap around
            overflow = (ptr + batch_size) - self.memory_bank_size
            self.memory_bank[ptr:] = embeddings[:batch_size-overflow].detach()
            self.memory_labels[ptr:] = labels[:batch_size-overflow]
            self.memory_bank[:overflow] = embeddings[batch_size-overflow:].detach()
            self.memory_labels[:overflow] = labels[batch_size-overflow:]

        # Update pointer
        ptr = (ptr + batch_size) % self.memory_bank_size
        self.memory_ptr[0] = ptr

    @torch.no_grad()
    def sample_from_memory(
        self,
        n_samples: int,
        label: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample negative examples from memory bank

        Args:
            n_samples: Number of samples to draw
            label: Optional label to filter by

        Returns:
            Sampled embeddings [n_samples, d]
        """
        if label is not None:
            # Sample from specific label
            mask = self.memory_labels == label
            valid_indices = torch.where(mask)[0]

            if len(valid_indices) < n_samples:
                # Not enough samples, use random
                indices = torch.randint(
                    0,
                    self.memory_bank_size,
                    (n_samples,),
                    device=self.memory_bank.device
                )
            else:
                # Random sample from valid indices
                perm = torch.randperm(len(valid_indices))
                indices = valid_indices[perm[:n_samples]]
        else:
            # Random sample
            indices = torch.randint(
                0,
                self.memory_bank_size,
                (n_samples,),
                device=self.memory_bank.device
            )

        return self.memory_bank[indices]


class InfoNCELoss(nn.Module):
    """InfoNCE loss variant"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss

        Args:
            anchor: Anchor embedding [B, d]
            positive: Positive embedding [B, d]
            negatives: Negative embeddings [B, N, d] or [N, d]

        Returns:
            Loss value
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(0)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # Negative similarity
        neg_sim = torch.bmm(
            negatives,
            anchor.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # Logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # Loss
        labels = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=logits.device
        )
        loss = F.cross_entropy(logits, labels)

        return loss
