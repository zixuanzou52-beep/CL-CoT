# CL-CoT: Contrastive Learning-Enhanced Chain-of-Thought for Complex Table QA

Official implementation of "Contrastive Learning-Enhanced Chain-of-Thought Optimization for Complex Table Question Answering".

## Overview

CL-CoT is a novel framework that enhances Chain-of-Thought reasoning for complex table question answering through:

- **Contrastive Learning**: Learn to distinguish high-quality from low-quality reasoning paths
- **Hierarchical Path Encoding**: Capture both step-level and path-level semantics
- **Reinforcement Learning**: Optimize template selection adaptively
- **Multi-dimensional Similarity**: Evaluate paths using structural, semantic, and operational dimensions

## Project Structure

```
CL-CoT/
├── configs/           # Configuration files
├── data/             # Data processing modules
├── models/           # Model implementations
├── training/         # Training modules
├── evaluation/       # Evaluation metrics
├── scripts/          # Training and evaluation scripts
├── utils/            # Utility functions
└── README.md         # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 40GB+ GPU memory for 13B model (or 24GB+ for 7B model)

### Setup

```bash
# Clone the repository
git clone https://github.com/zixuanzou52-beep/CL-CoT.git
cd CL-CoT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## Quick Start

### Data Preparation

Prepare your data in the following JSON format:

```json
{
  "table": {
    "headers": ["Year", "Revenue", "Profit"],
    "rows": [
      ["2019", "1000000", "200000"],
      ["2020", "1250000", "300000"]
    ]
  },
  "question": "What is the total revenue in 2020?",
  "answer": "1250000"
}
```

Place your data files in:
```
data/processed/
├── wtq/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── tabfact/
│   └── ...
└── hybridqa/
    └── ...
```

### Training

#### Stage 1: Supervised Pre-training

```bash
python scripts/train_stage1.py \
    --dataset wtq \
    --data_dir data/processed \
    --output_dir experiments/stage1/wtq \
    --num_epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5
```

#### Stage 2: Contrastive Learning 

First generate negative samples, then train:

```bash
# Generate negatives
python training/negative_generator.py \
    --input data/processed/wtq/train.json \
    --output data/processed/wtq/train_with_negatives.json

# Train Stage 2
python scripts/train_stage2.py \
    --pretrained_model experiments/stage1/wtq/best \
    --output_dir experiments/stage2/wtq
```

#### Stage 3: RL Fine-tuning 

```bash
python scripts/train_stage3.py \
    --pretrained_model experiments/stage2/wtq/best \
    --output_dir experiments/stage3/wtq
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model_path experiments/stage1/wtq/best \
    --dataset wtq \
    --split test \
    --output_file results/wtq_test.json
```

## Configuration

All configurations are in `configs/default.yaml`. Key parameters:

```yaml
# Model
model:
  base_model: "meta-llama/Llama-2-13b-hf"
  lora_rank: 16
  lora_alpha: 32

# Training
training:
  stage1_lr: 2e-5
  stage1_epochs: 3
  stage1_batch_size: 32

# Contrastive Learning
contrastive:
  temperature: 0.07
  negative_ratio: 5

# RL
rl:
  gamma: 0.95
  ppo_epsilon: 0.2
```

## Datasets

Supported datasets:

1. **WikiTableQuestions (WTQ)**: Complex compositional reasoning over Wikipedia tables
2. **TabFact**: Table-based fact verification
3. **HybridQA**: Multi-hop reasoning combining tables and text

Download links:
- WTQ: https://github.com/ppasupat/WikiTableQuestions
- TabFact: https://github.com/wenhuchen/Table-Fact-Checking
- HybridQA: https://github.com/wenhuchen/HybridQA

## Results

### WikiTableQuestions

| Model | EM (%) | F1 (%) | Avg Steps | Time (s) |
|-------|--------|--------|-----------|----------|
| CL-CoT | **62.8** | **68.9** | **6.8** | **1.82** |
| ReAcTable | 57.2 | 63.4 | 10.5 | 3.10 |
| Chain-of-Table | 55.8 | 62.1 | 9.8 | 2.95 |

(See paper for full results on all datasets)

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_path_encoder.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Quality

```bash
# Linting
flake8 . --max-line-length=120

# Type checking
mypy models/ --ignore-missing-imports

# Formatting
black . --line-length=120
```

## License

MIT License

## Contact

For questions and feedback:
- Email: zixuanzou52@gmail.com
- Issues: https://github.com/zixuanzou52-beep/CL-CoT/issues

## Acknowledgments

This work builds upon:
- Llama 2 by Meta AI
- LoRA by Microsoft
- Transformers by Hugging Face

---

**Note**: This is a research prototype. For production use, additional engineering and testing are recommended.
