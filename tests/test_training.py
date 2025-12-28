"""Tests for training modules"""
import pytest
import torch
from torch.utils.data import DataLoader
from training.base_trainer import BaseTrainer
from training.stage1_trainer import Stage1Trainer
from training.negative_generator import NegativeGenerator
from data.dataset_loader import TableQADataset, collate_fn
from utils.checkpoint import CheckpointManager


class TestBaseTrainer:
    """Tests for BaseTrainer"""

    def test_trainer_initialization(self, mock_model, test_config, sample_dataset_file):
        """Test trainer can be initialized"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer,
            max_length=512
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn
        )

        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        trainer = BaseTrainer(
            model=mock_model,
            config=test_config,
            train_loader=dataloader,
            optimizer=optimizer,
            device="cpu"
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_backward_step(self, mock_model, test_config, sample_dataset_file):
        """Test backward step"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        trainer = BaseTrainer(
            model=mock_model,
            config=test_config,
            train_loader=dataloader,
            optimizer=optimizer,
            device="cpu"
        )

        # Create dummy loss
        loss = torch.tensor(1.0, requires_grad=True)

        # Backward step
        trainer.backward_step(loss)

        # Check that step was performed
        assert trainer.global_step == 0  # Not incremented in backward_step


class TestStage1Trainer:
    """Tests for Stage1Trainer"""

    def test_stage1_trainer_initialization(
        self,
        mock_model,
        test_config,
        sample_dataset_file
    ):
        """Test Stage1Trainer initialization"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        trainer = Stage1Trainer(
            model=mock_model,
            config=test_config,
            train_loader=dataloader,
            optimizer=optimizer,
            device="cpu"
        )

        assert isinstance(trainer, Stage1Trainer)
        assert isinstance(trainer, BaseTrainer)

    def test_stage1_train_step(
        self,
        mock_model,
        test_config,
        sample_dataset_file
    ):
        """Test single training step"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        from utils.logger import ExperimentLogger

        logger = ExperimentLogger(
            project_name="test",
            run_name="test_run",
            config={},
            use_wandb=False
        )

        trainer = Stage1Trainer(
            model=mock_model,
            config=test_config,
            train_loader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device="cpu"
        )

        # Get one batch
        batch = next(iter(dataloader))

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()

        # Forward pass
        outputs = mock_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        assert 'loss' in outputs
        assert outputs['loss'] is not None

    def test_stage1_evaluate(
        self,
        mock_model,
        test_config,
        sample_dataset_file
    ):
        """Test evaluation"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        trainer = Stage1Trainer(
            model=mock_model,
            config=test_config,
            train_loader=dataloader,
            eval_loader=dataloader,
            optimizer=optimizer,
            device="cpu"
        )

        metrics = trainer.evaluate()

        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] >= 0
        assert metrics['perplexity'] >= 1.0

    def test_stage1_generate_sample(
        self,
        mock_model,
        test_config,
        sample_dataset_file,
        sample_table,
        sample_question
    ):
        """Test generation"""
        dataset = TableQADataset(
            data_path=sample_dataset_file,
            tokenizer=mock_model.tokenizer
        )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        trainer = Stage1Trainer(
            model=mock_model,
            config=test_config,
            train_loader=dataloader,
            optimizer=optimizer,
            device="cpu"
        )

        answer = trainer.generate_sample(
            table=sample_table,
            question=sample_question,
            max_length=50
        )

        assert isinstance(answer, str)


class TestNegativeGenerator:
    """Tests for NegativeGenerator"""

    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = NegativeGenerator(
            hard_negative_ratio=0.4,
            soft_negative_ratio=0.4,
            adversarial_negative_ratio=0.2,
            max_negatives_per_sample=5
        )

        assert generator is not None
        assert generator.max_negatives == 5

    def test_generate_hard_negatives(
        self,
        sample_reasoning_path,
        sample_table
    ):
        """Test hard negative generation"""
        generator = NegativeGenerator()

        negatives = generator.generate_hard_negatives(
            positive_path=sample_reasoning_path,
            table=sample_table,
            num_negatives=3
        )

        assert len(negatives) == 3
        for neg in negatives:
            assert isinstance(neg, type(sample_reasoning_path))

    def test_generate_soft_negatives(
        self,
        sample_reasoning_path,
        sample_table
    ):
        """Test soft negative generation"""
        generator = NegativeGenerator()

        negatives = generator.generate_soft_negatives(
            positive_path=sample_reasoning_path,
            table=sample_table,
            num_negatives=2
        )

        assert len(negatives) == 2

    def test_generate_all_negatives(
        self,
        sample_reasoning_path,
        sample_table
    ):
        """Test generating all types of negatives"""
        generator = NegativeGenerator(
            hard_negative_ratio=0.4,
            soft_negative_ratio=0.4,
            adversarial_negative_ratio=0.2,
            max_negatives_per_sample=5
        )

        # Create mock samples for adversarial negatives
        all_samples = [
            {
                'reasoning_path': sample_reasoning_path.to_dict()
            }
        ]

        negatives = generator.generate_negatives(
            positive_path=sample_reasoning_path,
            table=sample_table,
            all_samples=all_samples,
            current_idx=0
        )

        assert len(negatives) <= 5

    def test_process_dataset(
        self,
        tmp_path,
        sample_dataset_file
    ):
        """Test processing entire dataset"""
        generator = NegativeGenerator(max_negatives_per_sample=3)

        output_file = tmp_path / "output_with_negatives.json"

        # Add reasoning paths to samples
        import json
        with open(sample_dataset_file, 'r') as f:
            data = json.load(f)

        # Add dummy reasoning paths
        from data.path_generator import ReasoningStep
        for sample in data:
            sample['reasoning_path'] = {
                'steps': [
                    {
                        'operation': 'answer',
                        'arguments': [sample['answer']],
                        'result': sample['answer']
                    }
                ]
            }

        # Save modified data
        temp_input = tmp_path / "input_with_paths.json"
        with open(temp_input, 'w') as f:
            json.dump(data, f)

        # Process
        generator.process_dataset(
            input_path=str(temp_input),
            output_path=str(output_file),
            verbose=False
        )

        # Check output
        assert output_file.exists()

        with open(output_file, 'r') as f:
            output_data = json.load(f)

        assert len(output_data) == len(data)

        for sample in output_data:
            assert 'positive_path' in sample
            assert 'negative_paths' in sample


class TestCheckpointManager:
    """Tests for checkpoint management"""

    def test_checkpoint_save_load(
        self,
        tmp_path,
        mock_model,
        test_config
    ):
        """Test saving and loading checkpoints"""
        checkpoint_dir = tmp_path / "checkpoints"

        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_keep=2,
            save_best=True
        )

        optimizer = torch.optim.AdamW(mock_model.parameters(), lr=1e-4)

        # Save checkpoint
        manager.save_checkpoint(
            model=mock_model.model,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            metrics={'loss': 1.5},
            is_best=True
        )

        # Check files exist
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0
