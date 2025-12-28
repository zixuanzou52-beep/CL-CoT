"""PyTest configuration and fixtures"""
import pytest
import torch
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_model import CLCoTModel
from models.path_encoder import HierarchicalPathEncoder
from models.contrastive_loss import ContrastiveLoss
from models.similarity import MultiDimensionalSimilarity
from models.reward_function import RewardFunction
from data.table_parser import TableParser, Table
from data.path_generator import ReasoningPath, ReasoningStep
from utils.config import Config


@pytest.fixture(scope="session")
def device():
    """Get device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration"""
    config_dict = {
        'model': {
            'base_model': 'gpt2',  # Use smaller model for testing
            'use_lora': True,
            'lora_rank': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05
        },
        'encoder': {
            'hidden_dim': 768,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        },
        'contrastive': {
            'temperature': 0.07,
            'memory_bank_size': 100,
            'negative_ratio': 3
        },
        'training': {
            'stage1_batch_size': 2,
            'stage1_lr': 1e-4,
            'max_grad_norm': 1.0
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_dict, f)
        config_path = f.name

    config = Config(config_path)
    yield config

    # Cleanup
    Path(config_path).unlink()


@pytest.fixture
def sample_table():
    """Create sample table for testing"""
    return Table(
        table_id="test_table_1",
        headers=["Year", "Revenue", "Profit", "Employees"],
        rows=[
            ["2019", "1000000", "200000", "50"],
            ["2020", "1250000", "300000", "65"],
            ["2021", "1500000", "350000", "80"],
            ["2022", "1800000", "400000", "100"]
        ],
        types=["text", "number", "number", "number"]
    )


@pytest.fixture
def sample_question():
    """Sample question for testing"""
    return "What was the total revenue in 2020?"


@pytest.fixture
def sample_answer():
    """Sample answer for testing"""
    return "1250000"


@pytest.fixture
def sample_reasoning_path():
    """Create sample reasoning path"""
    steps = [
        ReasoningStep(
            operation="filter",
            arguments=["Year", "==", "2020"],
            result="Filtered to 2020 row"
        ),
        ReasoningStep(
            operation="select",
            arguments=["Revenue"],
            result="Selected Revenue column"
        ),
        ReasoningStep(
            operation="answer",
            arguments=["1250000"],
            result="1250000"
        )
    ]

    return ReasoningPath(steps=steps)


@pytest.fixture
def sample_dataset_file(sample_table, sample_question, sample_answer):
    """Create sample dataset file"""
    data = [
        {
            'table': {
                'headers': sample_table.headers,
                'rows': sample_table.rows
            },
            'question': sample_question,
            'answer': sample_answer,
            'table_id': 'test_1'
        },
        {
            'table': {
                'headers': ["Name", "Age", "City"],
                'rows': [
                    ["Alice", "30", "NYC"],
                    ["Bob", "25", "LA"]
                ]
            },
            'question': "How old is Alice?",
            'answer': "30",
            'table_id': 'test_2'
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        file_path = f.name

    yield file_path

    # Cleanup
    Path(file_path).unlink()


@pytest.fixture
def mock_model(device, test_config):
    """Create mock model for testing"""
    # Use very small model for testing
    model = CLCoTModel(
        model_name=test_config.get('model.base_model'),
        use_lora=False,  # Disable LoRA for faster testing
        device=device
    )
    return model


@pytest.fixture
def path_encoder(device, test_config):
    """Create path encoder"""
    encoder = HierarchicalPathEncoder(
        hidden_dim=test_config.get('encoder.hidden_dim'),
        num_layers=test_config.get('encoder.num_layers'),
        num_heads=test_config.get('encoder.num_heads'),
        dropout=test_config.get('encoder.dropout')
    ).to(device)
    return encoder


@pytest.fixture
def contrastive_loss(device, test_config):
    """Create contrastive loss"""
    loss_fn = ContrastiveLoss(
        temperature=test_config.get('contrastive.temperature'),
        memory_bank_size=test_config.get('contrastive.memory_bank_size')
    ).to(device)
    return loss_fn


@pytest.fixture
def similarity_function():
    """Create similarity function"""
    return MultiDimensionalSimilarity(
        struct_weight=0.3,
        semantic_weight=0.5,
        op_weight=0.2
    )


@pytest.fixture
def reward_function():
    """Create reward function"""
    return RewardFunction(
        penalty_coef=0.5,
        eff_weight=0.3,
        int_weight=0.2,
        max_steps=15
    )


@pytest.fixture
def table_parser():
    """Create table parser"""
    return TableParser()


# Helper functions for tests

def create_mock_embeddings(batch_size: int, dim: int, device: str = "cpu"):
    """Create mock embeddings for testing"""
    return torch.randn(batch_size, dim).to(device)


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert tensor has expected shape"""
    assert tensor.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_loss_valid(loss: torch.Tensor):
    """Assert loss is valid (scalar, finite, positive)"""
    assert loss.dim() == 0 or (loss.dim() == 1 and loss.size(0) == 1), \
        "Loss should be scalar"
    assert torch.isfinite(loss).all(), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"
