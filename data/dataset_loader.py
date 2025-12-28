"""Dataset loaders for table QA tasks"""
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional
from .table_parser import TableParser, Table
from .path_generator import ReasoningPath


class TableQADataset(Dataset):
    """Dataset for Table Question Answering"""

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512,
        include_paths: bool = False,
        parser_config: Optional[Dict] = None
    ):
        """
        Initialize dataset

        Args:
            data_path: Path to data file (JSON)
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            include_paths: Whether data includes reasoning paths
            parser_config: Configuration for TableParser
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_paths = include_paths

        # Initialize table parser
        parser_config = parser_config or {}
        self.parser = TableParser(**parser_config)

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different data formats
        if isinstance(data, dict):
            # May be nested under a key
            if 'data' in data:
                data = data['data']
            elif 'examples' in data:
                data = data['examples']

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        item = self.data[idx]

        # Parse table
        table = self.parser.parse(
            item['table'],
            table_id=item.get('table_id', str(idx))
        )

        # Get question and answer
        question = item['question']
        answer = item.get('answer', item.get('label', ''))

        # Build input text
        table_text = self.parser.linearize(table, format="markdown")
        input_text = f"Table:\n{table_text}\n\nQuestion: {question}\n\nAnswer:"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'table': table,
            'question': question,
            'answer': answer,
            'table_id': table.table_id
        }

        # Include reasoning path if available
        if self.include_paths and 'reasoning_path' in item:
            path_data = item['reasoning_path']
            if isinstance(path_data, dict):
                result['reasoning_path'] = ReasoningPath.from_dict(path_data)
            else:
                result['reasoning_path'] = path_data

        return result


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with positive and negative paths"""

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512,
        negative_ratio: int = 5
    ):
        """
        Initialize contrastive dataset

        Args:
            data_path: Path to data file with negatives
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            negative_ratio: Number of negatives per positive
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_ratio = negative_ratio

        self.parser = TableParser()
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data with negatives"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get contrastive sample"""
        item = self.data[idx]

        # Parse table
        table = self.parser.parse(item['table'])

        # Get question
        question = item['question']

        # Get positive path
        positive_path = ReasoningPath.from_dict(item['positive_path'])

        # Get negative paths
        negative_paths = []
        for neg_data in item.get('negative_paths', [])[:self.negative_ratio]:
            negative_paths.append(ReasoningPath.from_dict(neg_data))

        return {
            'table': table,
            'question': question,
            'positive_path': positive_path,
            'negative_paths': negative_paths,
            'answer': item.get('answer', '')
        }


def load_dataset(
    dataset_name: str,
    split: str,
    tokenizer: Any,
    data_dir: str = "data/processed",
    **kwargs
) -> Dataset:
    """
    Load dataset by name and split

    Args:
        dataset_name: Dataset name ('wtq', 'tabfact', 'hybridqa')
        split: Data split ('train', 'dev', 'test')
        tokenizer: Tokenizer
        data_dir: Data directory
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset object
    """
    data_path = Path(data_dir) / dataset_name / f"{split}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Determine dataset type based on file content
    with open(data_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
        if isinstance(sample, list):
            sample = sample[0]

    # Check if it has negative samples
    if 'negative_paths' in sample:
        return ContrastiveDataset(
            data_path=str(data_path),
            tokenizer=tokenizer,
            **kwargs
        )
    else:
        return TableQADataset(
            data_path=str(data_path),
            tokenizer=tokenizer,
            **kwargs
        )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader

    Args:
        batch: List of samples

    Returns:
        Batched data
    """
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    # Collect non-tensor data
    tables = [item['table'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'tables': tables,
        'questions': questions,
        'answers': answers
    }

    # Include paths if available
    if 'reasoning_path' in batch[0]:
        result['reasoning_paths'] = [item['reasoning_path'] for item in batch]

    # Include negatives if available
    if 'negative_paths' in batch[0]:
        result['positive_paths'] = [item['positive_path'] for item in batch]
        result['negative_paths'] = [item['negative_paths'] for item in batch]

    return result


class DatasetStatistics:
    """Compute and display dataset statistics"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def compute_stats(self) -> Dict[str, Any]:
        """Compute dataset statistics"""
        stats = {
            'num_samples': len(self.dataset),
            'avg_table_rows': 0,
            'avg_table_cols': 0,
            'avg_question_length': 0,
            'question_types': {}
        }

        total_rows = 0
        total_cols = 0
        total_q_len = 0

        for i in range(len(self.dataset)):
            item = self.dataset[i]

            # Table stats
            table = item['table']
            total_rows += len(table)
            total_cols += len(table.headers)

            # Question stats
            q_len = len(item['question'].split())
            total_q_len += q_len

        stats['avg_table_rows'] = total_rows / len(self.dataset)
        stats['avg_table_cols'] = total_cols / len(self.dataset)
        stats['avg_question_length'] = total_q_len / len(self.dataset)

        return stats

    def print_stats(self):
        """Print statistics"""
        stats = self.compute_stats()

        print("=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        print(f"Number of samples: {stats['num_samples']}")
        print(f"Average table rows: {stats['avg_table_rows']:.2f}")
        print(f"Average table columns: {stats['avg_table_cols']:.2f}")
        print(f"Average question length: {stats['avg_question_length']:.2f} words")
        print("=" * 50)
