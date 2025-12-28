"""Tests for contrastive loss"""
import pytest
import torch
from models.contrastive_loss import ContrastiveLoss, InfoNCELoss


class TestContrastiveLoss:
    """Tests for ContrastiveLoss"""

    def test_loss_initialization(self, contrastive_loss):
        """Test loss can be initialized"""
        assert contrastive_loss is not None
        assert hasattr(contrastive_loss, 'temperature')
        assert hasattr(contrastive_loss, 'memory_bank')

    def test_forward_single_sample(self, contrastive_loss, device):
        """Test forward pass with single sample"""
        dim = 768

        query_emb = torch.randn(dim).to(device)
        positive_emb = torch.randn(dim).to(device)
        negative_embs = torch.randn(5, dim).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_forward_batch(self, contrastive_loss, device):
        """Test forward pass with batch"""
        batch_size = 4
        num_negatives = 5
        dim = 768

        query_emb = torch.randn(batch_size, dim).to(device)
        positive_emb = torch.randn(batch_size, dim).to(device)
        negative_embs = torch.randn(batch_size, num_negatives, dim).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_temperature_effect(self, device):
        """Test temperature parameter effect"""
        dim = 768

        query_emb = torch.randn(dim).to(device)
        positive_emb = torch.randn(dim).to(device)
        negative_embs = torch.randn(5, dim).to(device)

        # Low temperature
        loss_fn_low = ContrastiveLoss(temperature=0.01).to(device)
        loss_low = loss_fn_low(query_emb, positive_emb, negative_embs)

        # High temperature
        loss_fn_high = ContrastiveLoss(temperature=1.0).to(device)
        loss_high = loss_fn_high(query_emb, positive_emb, negative_embs)

        # Losses should be different
        assert not torch.isclose(loss_low, loss_high, rtol=0.1)

    def test_memory_bank_update(self, contrastive_loss, device):
        """Test memory bank update"""
        batch_size = 8
        dim = 768

        embeddings = torch.randn(batch_size, dim).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # Get initial memory bank state
        initial_memory = contrastive_loss.memory_bank.clone()

        # Update memory bank
        contrastive_loss.update_memory_bank(embeddings, labels)

        # Memory bank should have changed
        assert not torch.allclose(
            contrastive_loss.memory_bank,
            initial_memory
        )

    def test_memory_bank_sampling(self, contrastive_loss, device):
        """Test sampling from memory bank"""
        # Populate memory bank
        batch_size = 10
        dim = 768

        embeddings = torch.randn(batch_size, dim).to(device)
        labels = torch.randint(0, 3, (batch_size,)).to(device)

        contrastive_loss.update_memory_bank(embeddings, labels)

        # Sample
        samples = contrastive_loss.sample_from_memory(n_samples=5)

        assert samples.shape == (5, dim)
        assert torch.isfinite(samples).all()

    def test_gradient_flow(self, contrastive_loss, device):
        """Test gradients can flow through loss"""
        dim = 768

        query_emb = torch.randn(dim, requires_grad=True).to(device)
        positive_emb = torch.randn(dim, requires_grad=True).to(device)
        negative_embs = torch.randn(5, dim, requires_grad=True).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        loss.backward()

        # Check gradients
        assert query_emb.grad is not None
        assert positive_emb.grad is not None
        assert negative_embs.grad is not None
        assert not torch.isnan(query_emb.grad).any()

    def test_perfect_match_loss(self, contrastive_loss, device):
        """Test loss when positive is identical to query"""
        dim = 768

        query_emb = torch.randn(dim).to(device)
        positive_emb = query_emb.clone()  # Perfect match
        negative_embs = torch.randn(5, dim).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        # Loss should be low (but not necessarily zero due to normalization)
        assert loss.item() < 2.0


class TestInfoNCELoss:
    """Tests for InfoNCELoss"""

    def test_infonce_initialization(self):
        """Test InfoNCE loss initialization"""
        loss_fn = InfoNCELoss(temperature=0.07)
        assert loss_fn is not None

    def test_infonce_forward(self, device):
        """Test InfoNCE forward pass"""
        loss_fn = InfoNCELoss(temperature=0.07)

        batch_size = 4
        num_negatives = 5
        dim = 768

        anchor = torch.randn(batch_size, dim).to(device)
        positive = torch.randn(batch_size, dim).to(device)
        negatives = torch.randn(batch_size, num_negatives, dim).to(device)

        loss = loss_fn(anchor, positive, negatives)

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_infonce_2d_negatives(self, device):
        """Test InfoNCE with 2D negatives"""
        loss_fn = InfoNCELoss(temperature=0.07)

        dim = 768

        anchor = torch.randn(1, dim).to(device)
        positive = torch.randn(1, dim).to(device)
        negatives = torch.randn(5, dim).to(device)  # 2D instead of 3D

        loss = loss_fn(anchor, positive, negatives)

        assert torch.isfinite(loss)

    def test_infonce_gradient_flow(self, device):
        """Test gradient flow through InfoNCE"""
        loss_fn = InfoNCELoss(temperature=0.07)

        anchor = torch.randn(2, 768, requires_grad=True).to(device)
        positive = torch.randn(2, 768, requires_grad=True).to(device)
        negatives = torch.randn(2, 5, 768, requires_grad=True).to(device)

        loss = loss_fn(anchor, positive, negatives)
        loss.backward()

        assert anchor.grad is not None
        assert not torch.isnan(anchor.grad).any()


class TestContrastiveLossEdgeCases:
    """Tests for edge cases"""

    def test_single_negative(self, contrastive_loss, device):
        """Test with single negative"""
        dim = 768

        query_emb = torch.randn(dim).to(device)
        positive_emb = torch.randn(dim).to(device)
        negative_embs = torch.randn(1, dim).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        assert torch.isfinite(loss)

    def test_many_negatives(self, contrastive_loss, device):
        """Test with many negatives"""
        dim = 768

        query_emb = torch.randn(dim).to(device)
        positive_emb = torch.randn(dim).to(device)
        negative_embs = torch.randn(20, dim).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        assert torch.isfinite(loss)

    def test_normalized_embeddings(self, contrastive_loss, device):
        """Test with pre-normalized embeddings"""
        import torch.nn.functional as F

        dim = 768

        query_emb = F.normalize(torch.randn(dim), dim=0).to(device)
        positive_emb = F.normalize(torch.randn(dim), dim=0).to(device)
        negative_embs = F.normalize(torch.randn(5, dim), dim=1).to(device)

        loss = contrastive_loss(
            query_emb=query_emb,
            positive_emb=positive_emb,
            negative_embs=negative_embs
        )

        assert torch.isfinite(loss)
