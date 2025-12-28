"""Tests for hierarchical path encoder"""
import pytest
import torch
from models.path_encoder import HierarchicalPathEncoder, SinusoidalPositionalEncoding, TextEncoder


class TestSinusoidalPositionalEncoding:
    """Tests for positional encoding"""

    def test_positional_encoding_shape(self, device):
        """Test positional encoding output shape"""
        d_model = 768
        max_len = 512
        batch_size = 2
        seq_len = 10

        pe = SinusoidalPositionalEncoding(d_model, max_len).to(device)

        x = torch.randn(batch_size, seq_len, d_model).to(device)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_values(self, device):
        """Test positional encoding adds to input"""
        d_model = 64
        pe = SinusoidalPositionalEncoding(d_model, 100).to(device)

        x = torch.zeros(1, 10, d_model).to(device)
        output = pe(x)

        # Output should not be all zeros
        assert not torch.allclose(output, x)
        assert torch.isfinite(output).all()


class TestHierarchicalPathEncoder:
    """Tests for hierarchical path encoder"""

    def test_encoder_initialization(self, path_encoder):
        """Test encoder can be initialized"""
        assert path_encoder is not None
        assert hasattr(path_encoder, 'step_encoder')
        assert hasattr(path_encoder, 'path_attention')

    def test_encode_step(self, path_encoder, device):
        """Test encoding a single step"""
        hidden_dim = path_encoder.hidden_dim
        seq_len = 10

        step_emb = torch.randn(seq_len, hidden_dim).to(device)
        table_emb = torch.randn(hidden_dim).to(device)
        question_emb = torch.randn(hidden_dim).to(device)

        step_repr = path_encoder.encode_step(
            step_emb=step_emb,
            table_emb=table_emb,
            question_emb=question_emb
        )

        assert step_repr.shape == (hidden_dim,)
        assert torch.isfinite(step_repr).all()

    def test_encode_path(self, path_encoder, device):
        """Test encoding complete path"""
        hidden_dim = path_encoder.hidden_dim
        num_steps = 5

        step_reprs = torch.randn(num_steps, hidden_dim).to(device)
        question_emb = torch.randn(hidden_dim).to(device)

        path_repr = path_encoder.encode_path(
            step_reprs=step_reprs,
            question_emb=question_emb
        )

        assert path_repr.shape == (hidden_dim,)
        assert torch.isfinite(path_repr).all()

    def test_forward_pass(self, path_encoder, sample_reasoning_path, device):
        """Test full forward pass"""
        hidden_dim = path_encoder.hidden_dim

        table_emb = torch.randn(hidden_dim).to(device)
        question_emb = torch.randn(hidden_dim).to(device)

        # Create mock step embeddings
        step_embeddings = [
            torch.randn(10, hidden_dim).to(device)
            for _ in range(len(sample_reasoning_path.steps))
        ]

        output = path_encoder(
            reasoning_path=sample_reasoning_path,
            table_emb=table_emb,
            question_emb=question_emb,
            step_embeddings=step_embeddings
        )

        assert 'path_embedding' in output
        assert 'step_embeddings' in output

        assert output['path_embedding'].shape == (hidden_dim,)
        assert output['step_embeddings'].shape == (len(sample_reasoning_path.steps), hidden_dim)

    def test_different_num_steps(self, path_encoder, device):
        """Test with different number of steps"""
        hidden_dim = path_encoder.hidden_dim

        for num_steps in [1, 3, 5, 10]:
            step_reprs = torch.randn(num_steps, hidden_dim).to(device)
            question_emb = torch.randn(hidden_dim).to(device)

            path_repr = path_encoder.encode_path(
                step_reprs=step_reprs,
                question_emb=question_emb
            )

            assert path_repr.shape == (hidden_dim,)

    def test_gradient_flow(self, path_encoder, device):
        """Test gradients can flow through encoder"""
        hidden_dim = path_encoder.hidden_dim

        step_reprs = torch.randn(3, hidden_dim, requires_grad=True).to(device)
        question_emb = torch.randn(hidden_dim, requires_grad=True).to(device)

        path_repr = path_encoder.encode_path(
            step_reprs=step_reprs,
            question_emb=question_emb
        )

        # Backpropagate
        loss = path_repr.sum()
        loss.backward()

        # Check gradients exist
        assert step_reprs.grad is not None
        assert question_emb.grad is not None
        assert not torch.isnan(step_reprs.grad).any()


class TestTextEncoder:
    """Tests for text encoder"""

    def test_text_encoder_initialization(self, mock_model):
        """Test text encoder can be initialized"""
        text_encoder = TextEncoder(
            tokenizer=mock_model.tokenizer,
            model=mock_model.base_model,
            hidden_dim=768
        )

        assert text_encoder is not None

    def test_encode_single_text(self, mock_model, device):
        """Test encoding single text"""
        text_encoder = TextEncoder(
            tokenizer=mock_model.tokenizer,
            model=mock_model.base_model,
            hidden_dim=768
        )

        text = "This is a test sentence."
        embedding = text_encoder.encode(text, device=device)

        assert embedding.dim() == 1
        assert embedding.shape[0] == 768
        assert torch.isfinite(embedding).all()

    def test_encode_batch(self, mock_model, device):
        """Test encoding batch of texts"""
        text_encoder = TextEncoder(
            tokenizer=mock_model.tokenizer,
            model=mock_model.base_model,
            hidden_dim=768
        )

        texts = [
            "First sentence.",
            "Second sentence is longer.",
            "Third."
        ]

        embeddings = text_encoder.encode_batch(texts, device=device)

        assert embeddings.shape == (3, 768)
        assert torch.isfinite(embeddings).all()

    def test_encode_empty_text(self, mock_model, device):
        """Test encoding empty text"""
        text_encoder = TextEncoder(
            tokenizer=mock_model.tokenizer,
            model=mock_model.base_model,
            hidden_dim=768
        )

        # Should handle empty text gracefully
        embedding = text_encoder.encode("", device=device)
        assert embedding.shape[0] == 768

    def test_mean_pooling_with_mask(self, mock_model, device):
        """Test mean pooling respects attention mask"""
        text_encoder = TextEncoder(
            tokenizer=mock_model.tokenizer,
            model=mock_model.base_model,
            hidden_dim=768
        )

        # Different length texts
        texts = ["Short", "This is a much longer sentence"]
        embeddings = text_encoder.encode_batch(texts, device=device)

        # Should produce different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1])
