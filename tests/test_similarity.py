"""Tests for similarity functions"""
import pytest
import torch
from models.similarity import (
    MultiDimensionalSimilarity,
    StructuralSimilarity,
    SemanticSimilarity,
    OperationalSimilarity
)
from data.path_generator import ReasoningPath, ReasoningStep


class TestStructuralSimilarity:
    """Tests for structural similarity"""

    def test_identical_paths(self):
        """Test similarity of identical paths"""
        sim_fn = StructuralSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", ["Year", "==", "2020"], ""),
            ReasoningStep("select", ["Revenue"], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("filter", ["Year", "==", "2020"], ""),
            ReasoningStep("select", ["Revenue"], "")
        ])

        similarity = sim_fn.compute(path1, path2)

        assert similarity == 1.0

    def test_different_operations(self):
        """Test similarity with different operations"""
        sim_fn = StructuralSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("sort", [], ""),
            ReasoningStep("aggregate", [], "")
        ])

        similarity = sim_fn.compute(path1, path2)

        # Should be less than 1 since operations differ
        assert 0 <= similarity < 1.0

    def test_different_lengths(self):
        """Test similarity with different path lengths"""
        sim_fn = StructuralSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], ""),
            ReasoningStep("aggregate", [], "")
        ])

        similarity = sim_fn.compute(path1, path2)

        # Should penalize length difference
        assert 0 <= similarity < 1.0

    def test_empty_paths(self):
        """Test similarity of empty paths"""
        sim_fn = StructuralSimilarity()

        path1 = ReasoningPath(steps=[])
        path2 = ReasoningPath(steps=[])

        similarity = sim_fn.compute(path1, path2)

        assert similarity == 1.0


class TestSemanticSimilarity:
    """Tests for semantic similarity"""

    def test_semantic_similarity(self, device):
        """Test semantic similarity computation"""
        sim_fn = SemanticSimilarity()

        # Create mock embeddings
        emb1 = torch.randn(768).to(device)
        emb2 = torch.randn(768).to(device)

        similarity = sim_fn.compute_from_embeddings(emb1, emb2)

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

    def test_identical_embeddings(self, device):
        """Test similarity of identical embeddings"""
        sim_fn = SemanticSimilarity()

        emb = torch.randn(768).to(device)
        similarity = sim_fn.compute_from_embeddings(emb, emb)

        assert abs(similarity - 1.0) < 1e-5

    def test_orthogonal_embeddings(self, device):
        """Test similarity of orthogonal embeddings"""
        sim_fn = SemanticSimilarity()

        # Create orthogonal vectors
        emb1 = torch.zeros(768).to(device)
        emb1[0] = 1.0

        emb2 = torch.zeros(768).to(device)
        emb2[1] = 1.0

        similarity = sim_fn.compute_from_embeddings(emb1, emb2)

        assert abs(similarity) < 1e-5


class TestOperationalSimilarity:
    """Tests for operational similarity"""

    def test_operation_overlap(self):
        """Test operation set overlap"""
        sim_fn = OperationalSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], ""),
            ReasoningStep("aggregate", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], "")
        ])

        similarity = sim_fn.compute(path1, path2)

        # Both have filter and select, path1 has aggregate
        # Jaccard: 2 / 3 = 0.667
        assert 0.6 < similarity <= 0.7

    def test_no_overlap(self):
        """Test with no operation overlap"""
        sim_fn = OperationalSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("aggregate", [], "")
        ])

        similarity = sim_fn.compute(path1, path2)

        assert similarity == 0.0

    def test_complete_overlap(self):
        """Test with complete operation overlap"""
        sim_fn = OperationalSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("select", [], ""),
            ReasoningStep("filter", [], "")
        ])

        similarity = sim_fn.compute(path1, path2)

        assert similarity == 1.0


class TestMultiDimensionalSimilarity:
    """Tests for multi-dimensional similarity"""

    def test_initialization(self, similarity_function):
        """Test initialization"""
        assert similarity_function is not None
        assert hasattr(similarity_function, 'struct_weight')
        assert hasattr(similarity_function, 'semantic_weight')
        assert hasattr(similarity_function, 'op_weight')

    def test_weights_sum_to_one(self):
        """Test weights sum to 1"""
        sim_fn = MultiDimensionalSimilarity(
            struct_weight=0.3,
            semantic_weight=0.5,
            op_weight=0.2
        )

        total_weight = (
            sim_fn.struct_weight +
            sim_fn.semantic_weight +
            sim_fn.op_weight
        )

        assert abs(total_weight - 1.0) < 1e-6

    def test_compute_similarity(self, device):
        """Test computing multi-dimensional similarity"""
        sim_fn = MultiDimensionalSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", ["Year"], ""),
            ReasoningStep("select", ["Revenue"], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("filter", ["Year"], ""),
            ReasoningStep("select", ["Profit"], "")
        ])

        # Mock embeddings
        emb1 = torch.randn(768).to(device)
        emb2 = torch.randn(768).to(device)

        similarity = sim_fn.compute(
            path1=path1,
            path2=path2,
            emb1=emb1,
            emb2=emb2
        )

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1.0

    def test_structural_only(self):
        """Test with structural similarity only"""
        sim_fn = MultiDimensionalSimilarity(
            struct_weight=1.0,
            semantic_weight=0.0,
            op_weight=0.0
        )

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("filter", [], "")
        ])

        # Without embeddings (structural only)
        similarity = sim_fn.compute_structural_only(path1, path2)

        assert similarity > 0

    def test_identical_paths_high_similarity(self, device):
        """Test identical paths have high similarity"""
        sim_fn = MultiDimensionalSimilarity()

        path = ReasoningPath(steps=[
            ReasoningStep("filter", ["Year", "==", "2020"], ""),
            ReasoningStep("select", ["Revenue"], "")
        ])

        emb = torch.randn(768).to(device)

        similarity = sim_fn.compute(path, path, emb, emb)

        # Should be very high (close to 1.0)
        assert similarity > 0.9

    def test_different_paths_lower_similarity(self, device):
        """Test different paths have lower similarity"""
        sim_fn = MultiDimensionalSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], "")
        ])

        path2 = ReasoningPath(steps=[
            ReasoningStep("sort", [], ""),
            ReasoningStep("aggregate", [], ""),
            ReasoningStep("answer", [], "")
        ])

        emb1 = torch.randn(768).to(device)
        emb2 = torch.randn(768).to(device)

        similarity = sim_fn.compute(path1, path2, emb1, emb2)

        # Should be lower
        assert similarity < 0.9


class TestSimilarityEdgeCases:
    """Tests for edge cases"""

    def test_empty_path_similarity(self):
        """Test similarity with empty paths"""
        sim_fn = MultiDimensionalSimilarity()

        path1 = ReasoningPath(steps=[])
        path2 = ReasoningPath(steps=[])

        similarity = sim_fn.compute_structural_only(path1, path2)

        # Empty paths should be similar
        assert similarity == 1.0

    def test_one_empty_path(self):
        """Test similarity with one empty path"""
        sim_fn = MultiDimensionalSimilarity()

        path1 = ReasoningPath(steps=[
            ReasoningStep("filter", [], "")
        ])

        path2 = ReasoningPath(steps=[])

        similarity = sim_fn.compute_structural_only(path1, path2)

        # Should have low similarity
        assert similarity < 1.0

    def test_custom_weights(self):
        """Test with custom weights"""
        sim_fn = MultiDimensionalSimilarity(
            struct_weight=0.5,
            semantic_weight=0.3,
            op_weight=0.2
        )

        assert sim_fn.struct_weight == 0.5
        assert sim_fn.semantic_weight == 0.3
        assert sim_fn.op_weight == 0.2
