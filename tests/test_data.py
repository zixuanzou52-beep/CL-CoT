"""Tests for data loading and processing"""
import pytest
import torch
from pathlib import Path
from data.dataset_loader import TableQADataset, ContrastiveDataset, load_dataset, collate_fn
from data.table_parser import TableParser, Table
from data.path_generator import ReasoningPath, ReasoningStep


class TestTableParser:
    """Tests for TableParser"""

    def test_parse_table(self, table_parser):
        """Test table parsing"""
        table_dict = {
            'headers': ['Name', 'Age', 'City'],
            'rows': [
                ['Alice', '30', 'NYC'],
                ['Bob', '25', 'LA']
            ]
        }

        table = table_parser.parse(table_dict, table_id='test_1')

        assert isinstance(table, Table)
        assert table.table_id == 'test_1'
        assert table.headers == ['Name', 'Age', 'City']
        assert len(table.rows) == 2
        assert table.rows[0] == ['Alice', '30', 'NYC']

    def test_linearize_markdown(self, table_parser, sample_table):
        """Test table linearization to markdown"""
        text = table_parser.linearize(sample_table, format="markdown")

        assert isinstance(text, str)
        assert 'Year' in text
        assert 'Revenue' in text
        assert '2020' in text
        assert '1250000' in text
        assert '|' in text  # Markdown table separator

    def test_linearize_plain(self, table_parser, sample_table):
        """Test table linearization to plain text"""
        text = table_parser.linearize(sample_table, format="plain")

        assert isinstance(text, str)
        assert 'Year' in text
        assert 'Revenue' in text

    def test_infer_types(self, table_parser):
        """Test type inference"""
        table_dict = {
            'headers': ['Name', 'Age', 'Score'],
            'rows': [
                ['Alice', '30', '95.5'],
                ['Bob', '25', '87.3']
            ]
        }

        table = table_parser.parse(table_dict)

        assert 'text' in table.types[0].lower() or 'string' in table.types[0].lower()
        assert 'number' in table.types[1].lower() or 'int' in table.types[1].lower()


class TestReasoningPath:
    """Tests for ReasoningPath"""

    def test_create_path(self, sample_reasoning_path):
        """Test creating reasoning path"""
        assert isinstance(sample_reasoning_path, ReasoningPath)
        assert len(sample_reasoning_path.steps) == 3
        assert sample_reasoning_path.steps[0].operation == "filter"

    def test_path_to_dict(self, sample_reasoning_path):
        """Test converting path to dict"""
        path_dict = sample_reasoning_path.to_dict()

        assert isinstance(path_dict, dict)
        assert 'steps' in path_dict
        assert len(path_dict['steps']) == 3

    def test_path_from_dict(self, sample_reasoning_path):
        """Test creating path from dict"""
        path_dict = sample_reasoning_path.to_dict()
        reconstructed = ReasoningPath.from_dict(path_dict)

        assert len(reconstructed.steps) == len(sample_reasoning_path.steps)
        assert reconstructed.steps[0].operation == sample_reasoning_path.steps[0].operation

    def test_step_to_text(self):
        """Test step to text conversion"""
        step = ReasoningStep(
            operation="filter",
            arguments=["Year", "==", "2020"],
            result="Filtered to 2020"
        )

        text = step.to_text()
        assert isinstance(text, str)
        assert "filter" in text.lower()


class TestTableQADataset:
    """Tests for TableQADataset"""

    def test_load_dataset(self, sample_dataset_file, mock_model):
        """Test loading dataset"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer,
            max_length=512
        )

        assert len(dataset) == 2

    def test_getitem(self, sample_dataset_file, mock_model):
        """Test getting item from dataset"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer,
            max_length=512
        )

        item = dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'table' in item
        assert 'question' in item
        assert 'answer' in item

        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)

    def test_dataset_length(self, sample_dataset_file, mock_model):
        """Test dataset length"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        assert len(dataset) == 2


class TestCollateFunction:
    """Tests for collate function"""

    def test_collate_batch(self, sample_dataset_file, mock_model):
        """Test collating batch"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)

        assert 'input_ids' in collated
        assert 'attention_mask' in collated
        assert 'tables' in collated
        assert 'questions' in collated
        assert 'answers' in collated

        assert collated['input_ids'].shape[0] == 2
        assert len(collated['questions']) == 2

    def test_collate_single_item(self, sample_dataset_file, mock_model):
        """Test collating single item"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        batch = [dataset[0]]
        collated = collate_fn(batch)

        assert collated['input_ids'].shape[0] == 1


class TestLoadDataset:
    """Tests for load_dataset function"""

    def test_load_by_name(self, tmp_path, mock_model):
        """Test loading dataset by name"""
        # Create temporary dataset directory
        data_dir = tmp_path / "data"
        dataset_dir = data_dir / "wtq"
        dataset_dir.mkdir(parents=True)

        # Create train file
        import json
        train_data = [
            {
                'table': {
                    'headers': ['A', 'B'],
                    'rows': [['1', '2']]
                },
                'question': 'What is A?',
                'answer': '1'
            }
        ]

        train_file = dataset_dir / "train.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f)

        # Load dataset
        dataset = load_dataset(
            dataset_name='wtq',
            split='train',
            tokenizer=mock_model.tokenizer,
            data_dir=str(data_dir)
        )

        assert len(dataset) == 1

    def test_load_missing_file(self, tmp_path, mock_model):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_dataset(
                dataset_name='nonexistent',
                split='train',
                tokenizer=mock_model.tokenizer,
                data_dir=str(tmp_path)
            )
