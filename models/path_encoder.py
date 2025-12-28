"""Hierarchical path encoder for reasoning paths"""
import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        return x + self.pe[:, :x.size(1)]


class HierarchicalPathEncoder(nn.Module):
    """
    Hierarchical encoder for reasoning paths

    Encodes at two levels:
    1. Step-level: Encode individual reasoning steps
    2. Path-level: Aggregate step representations into path representation
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_position_embeddings: int = 512
    ):
        """
        Initialize hierarchical path encoder

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_position_embeddings: Maximum position embeddings
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding layers
        self.embedding = nn.Linear(768, hidden_dim)  # Assume 768-dim input

        # Positional encoding
        self.position_encoding = SinusoidalPositionalEncoding(
            hidden_dim,
            max_position_embeddings
        )

        # Step-level encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.step_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Path-level attention
        self.path_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def encode_step(
        self,
        step_emb: torch.Tensor,
        table_emb: torch.Tensor,
        question_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single reasoning step

        h_i = TransformerEncoder([s_i; v_T; v_Q])

        Args:
            step_emb: Step embedding [L, d]
            table_emb: Table embedding [d]
            question_emb: Question embedding [d]

        Returns:
            Step representation [d]
        """
        # Expand table and question embeddings
        table_emb = table_emb.unsqueeze(0)  # [1, d]
        question_emb = question_emb.unsqueeze(0)  # [1, d]

        # Concatenate: [step_tokens; table; question]
        combined = torch.cat([step_emb, table_emb, question_emb], dim=0)  # [L+2, d]

        # Add positional encoding
        combined = self.position_encoding(combined.unsqueeze(0))  # [1, L+2, d]

        # Transform encoder
        encoded = self.step_encoder(combined)  # [1, L+2, d]

        # Pool (mean pooling)
        step_repr = encoded.mean(dim=1).squeeze(0)  # [d]

        return step_repr

    def encode_path(
        self,
        step_reprs: torch.Tensor,
        question_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode complete reasoning path

        h_P = Attention(h_1, h_2, ..., h_k)

        Args:
            step_reprs: Step representations [K, d]
            question_emb: Question embedding [d]

        Returns:
            Path representation [d]
        """
        # Use question as query
        query = question_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
        key_value = step_reprs.unsqueeze(0)  # [1, K, d]

        # Multi-head attention
        path_repr, attn_weights = self.path_attention(
            query=query,
            key=key_value,
            value=key_value
        )  # [1, 1, d], [1, 1, K]

        path_repr = path_repr.squeeze(0).squeeze(0)  # [d]

        # Layer norm
        path_repr = self.layer_norm(path_repr)

        return path_repr

    def forward(
        self,
        reasoning_path: Any,
        table_emb: torch.Tensor,
        question_emb: torch.Tensor,
        step_embeddings: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            reasoning_path: ReasoningPath object
            table_emb: Table embedding [d]
            question_emb: Question embedding [d]
            step_embeddings: Pre-computed step embeddings (optional)

        Returns:
            Dictionary with path_embedding and step_embeddings
        """
        # If step embeddings not provided, they need to be computed
        # In practice, these would come from a separate text encoder
        if step_embeddings is None:
            # Placeholder - in real implementation, encode step text
            num_steps = len(reasoning_path.steps)
            step_embeddings = [
                torch.randn(10, self.hidden_dim).to(table_emb.device)
                for _ in range(num_steps)
            ]

        # Encode each step
        step_reprs = []
        for step_emb in step_embeddings:
            step_repr = self.encode_step(step_emb, table_emb, question_emb)
            step_reprs.append(step_repr)

        step_reprs = torch.stack(step_reprs)  # [K, d]

        # Encode path
        path_embedding = self.encode_path(step_reprs, question_emb)

        return {
            'path_embedding': path_embedding,
            'step_embeddings': step_reprs
        }


class TextEncoder(nn.Module):
    """Encode text using pretrained model"""

    def __init__(self, tokenizer, model, hidden_dim: int = 768):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.hidden_dim = hidden_dim

    def encode(
        self,
        text: str,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Encode text to embedding

        Args:
            text: Input text
            device: Device to use

        Returns:
            Text embedding [d]
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state, mean pool
            hidden_states = outputs.hidden_states[-1]  # [1, L, d]
            embedding = hidden_states.mean(dim=1).squeeze(0)  # [d]

        return embedding

    def encode_batch(
        self,
        texts: List[str],
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Encode batch of texts

        Args:
            texts: List of texts
            device: Device to use

        Returns:
            Batch of embeddings [B, d]
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [B, L, d]

            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [B, L, 1]
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [B, d]
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            embeddings = sum_hidden / sum_mask  # [B, d]

        return embeddings
