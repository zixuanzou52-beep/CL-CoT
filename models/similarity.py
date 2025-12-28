"""Multi-dimensional similarity computation"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiDimensionalSimilarity:
    """
    Compute multi-dimensional similarity between reasoning paths

    Three types of similarity:
    1. Structural similarity (based on LCS)
    2. Semantic similarity (based on embeddings)
    3. Operational similarity (based on Jaccard)
    """

    def __init__(
        self,
        struct_weight: float = 0.3,
        semantic_weight: float = 0.5,
        op_weight: float = 0.2
    ):
        """
        Initialize similarity calculator

        Args:
            struct_weight: Weight for structural similarity
            semantic_weight: Weight for semantic similarity
            op_weight: Weight for operational similarity
        """
        self.alpha = struct_weight
        self.beta = semantic_weight
        self.gamma = op_weight

        # Ensure weights sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

    def structural_similarity(
        self,
        path1: Any,
        path2: Any
    ) -> float:
        """
        Compute structural similarity using LCS

        Sim_struct(P1, P2) = LCS(P1, P2) / max(|P1|, |P2|)

        Args:
            path1: First reasoning path
            path2: Second reasoning path

        Returns:
            Structural similarity [0, 1]
        """
        steps1 = path1.steps
        steps2 = path2.steps

        # Compute LCS length
        lcs_length = self._longest_common_subsequence(steps1, steps2)

        # Normalize
        max_length = max(len(steps1), len(steps2))
        similarity = lcs_length / max_length if max_length > 0 else 0.0

        return similarity

    def semantic_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute semantic similarity using cosine similarity

        Sim_sem(P1, P2) = cos(h_P1, h_P2)

        Args:
            embedding1: First path embedding [d]
            embedding2: Second path embedding [d]

        Returns:
            Semantic similarity [-1, 1]
        """
        # Ensure tensors are 2D for F.cosine_similarity
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)

        cosine_sim = F.cosine_similarity(embedding1, embedding2)

        return cosine_sim.item()

    def operational_similarity(
        self,
        path1: Any,
        path2: Any
    ) -> float:
        """
        Compute operational similarity using Jaccard

        Sim_op(P1, P2) = |O_1 ∩ O_2| / |O_1 ∪ O_2|

        Args:
            path1: First reasoning path
            path2: Second reasoning path

        Returns:
            Operational similarity [0, 1]
        """
        ops1 = set(path1.operations)
        ops2 = set(path2.operations)

        intersection = len(ops1 & ops2)
        union = len(ops1 | ops2)

        jaccard = intersection / union if union > 0 else 0.0

        return jaccard

    def compute_similarity(
        self,
        path1: Any,
        path2: Any,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute total similarity

        Sim = α·Sim_struct + β·Sim_sem + γ·Sim_op

        Args:
            path1: First reasoning path
            path2: Second reasoning path
            emb1: First path embedding
            emb2: Second path embedding

        Returns:
            Dictionary with all similarities
        """
        struct_sim = self.structural_similarity(path1, path2)
        sem_sim = self.semantic_similarity(emb1, emb2)
        op_sim = self.operational_similarity(path1, path2)

        total_sim = (
            self.alpha * struct_sim +
            self.beta * sem_sim +
            self.gamma * op_sim
        )

        return {
            'structural': struct_sim,
            'semantic': sem_sim,
            'operational': op_sim,
            'total': total_sim
        }

    @staticmethod
    def _longest_common_subsequence(seq1: list, seq2: list) -> int:
        """
        Compute LCS length using dynamic programming

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            LCS length
        """
        m, n = len(seq1), len(seq2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]


def batch_cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor
) -> torch.Tensor:
    """
    Compute batch cosine similarity

    Args:
        embeddings1: First batch of embeddings [B1, d]
        embeddings2: Second batch of embeddings [B2, d]

    Returns:
        Similarity matrix [B1, B2]
    """
    # Normalize
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Compute similarity
    similarity = torch.matmul(embeddings1, embeddings2.t())

    return similarity
