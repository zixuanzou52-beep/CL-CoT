"""Tests for model components"""
import pytest
import torch
from models.base_model import CLCoTModel


class TestCLCoTModel:
    """Tests for CLCoTModel"""

    def test_model_initialization(self, test_config, device):
        """Test model can be initialized"""
        model = CLCoTModel(
            model_name=test_config.get('model.base_model'),
            use_lora=False,
            device=device
        )

        assert model is not None
        assert model.tokenizer is not None
        assert model.model is not None

    def test_model_forward_pass(self, mock_model, device):
        """Test forward pass"""
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        labels = input_ids.clone()

        outputs = mock_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        assert 'loss' in outputs
        assert 'logits' in outputs
        assert outputs['loss'] is not None
        assert outputs['logits'].shape[0] == batch_size
        assert outputs['logits'].shape[1] == seq_len

    def test_model_generate(self, mock_model, device):
        """Test generation"""
        input_text = "Question: What is 2+2? Answer:"
        inputs = mock_model.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(device)

        outputs = mock_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=20,
            do_sample=False
        )

        assert outputs.shape[0] == 1
        assert outputs.shape[1] <= 20

        # Decode
        generated_text = mock_model.tokenizer.decode(outputs[0])
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0

    def test_model_with_lora(self, test_config, device):
        """Test model with LoRA"""
        model = CLCoTModel(
            model_name=test_config.get('model.base_model'),
            use_lora=True,
            lora_rank=test_config.get('model.lora_rank'),
            lora_alpha=test_config.get('model.lora_alpha'),
            device=device
        )

        # Check trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())

        # With LoRA, trainable params should be much less than total
        assert trainable_params < total_params
        assert trainable_params > 0

    def test_model_save_and_load(self, mock_model, tmp_path):
        """Test model saving and loading"""
        save_dir = tmp_path / "test_model"

        # Save
        mock_model.save_pretrained(str(save_dir))

        # Check files exist
        assert save_dir.exists()
        assert (save_dir / "config.json").exists() or list(save_dir.glob("*.json"))

    def test_tokenizer_padding(self, mock_model):
        """Test tokenizer has padding token"""
        assert mock_model.tokenizer.pad_token is not None

    def test_model_device_placement(self, test_config):
        """Test model can be placed on different devices"""
        # CPU model
        model_cpu = CLCoTModel(
            model_name=test_config.get('model.base_model'),
            use_lora=False,
            device="cpu"
        )

        assert model_cpu.device == "cpu"

        # Test forward on CPU
        input_ids = torch.randint(0, 100, (1, 5))
        outputs = model_cpu(input_ids=input_ids)
        assert outputs['logits'].device.type == "cpu"

    def test_get_trainable_parameters(self, mock_model):
        """Test get trainable parameters count"""
        num_params = mock_model.get_trainable_parameters()
        assert isinstance(num_params, int)
        assert num_params > 0
