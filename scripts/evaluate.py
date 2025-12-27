"""Evaluation script for CL-CoT models"""
import argparse
import torch
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config
from models.base_model import CLCoTModel
from data.dataset_loader import load_dataset, collate_fn
from evaluation.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CL-CoT model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wtq', 'tabfact', 'hybridqa'],
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'dev', 'test'],
                       help='Data split to evaluate')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')

    return parser.parse_args()


def extract_answer(generated_text: str, question: str) -> str:
    """
    Extract answer from generated text

    Args:
        generated_text: Generated text from model
        question: Original question

    Returns:
        Extracted answer
    """
    # Remove the question part
    if question in generated_text:
        generated_text = generated_text.split(question)[-1]

    # Look for answer markers
    markers = ['Answer:', 'Final Answer:', 'Result:', '\nAnswer:']
    for marker in markers:
        if marker in generated_text:
            answer = generated_text.split(marker)[-1].strip()
            # Take first line
            answer = answer.split('\n')[0].strip()
            return answer

    # If no marker found, return last line
    lines = [l.strip() for l in generated_text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]

    return generated_text.strip()


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset

    Args:
        model: CL-CoT model
        dataloader: DataLoader
        device: Device to use

    Returns:
        Dictionary with predictions, ground truths, and times
    """
    model.eval()

    predictions = []
    ground_truths = []
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            questions = batch['questions']
            answers = batch['answers']

            # Measure inference time
            start_time = time.time()

            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )

            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(questions))

            # Decode predictions
            for i, output in enumerate(outputs):
                generated_text = model.tokenizer.decode(output, skip_special_tokens=True)
                prediction = extract_answer(generated_text, questions[i])
                predictions.append(prediction)
                ground_truths.append(answers[i])

    return {
        'predictions': predictions,
        'ground_truths': ground_truths,
        'inference_times': inference_times
    }


def main():
    args = parse_args()

    # Load config (if exists)
    config_path = Path(args.model_path) / 'config.yaml'
    if config_path.exists():
        config = Config(str(config_path))
    else:
        config = Config()

    print(f"Evaluating model: {args.model_path}")
    print(f"Dataset: {args.dataset}, Split: {args.split}")

    # Load model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLCoTModel.from_pretrained(args.model_path, device=device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        tokenizer=model.tokenizer,
        data_dir=args.data_dir
    )

    # Limit samples if specified
    if args.max_samples:
        dataset.data = dataset.data[:args.max_samples]

    print(f"Evaluating on {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Evaluate
    print("Running evaluation...")
    results = evaluate_model(model, dataloader, device)

    # Compute metrics
    print("Computing metrics...")
    calculator = MetricsCalculator()

    metrics = calculator.compute_all(
        results['predictions'],
        results['ground_truths']
    )

    # Add efficiency metrics
    avg_time = sum(results['inference_times']) / len(results['inference_times'])
    metrics['avg_inference_time'] = avg_time

    # Print metrics
    calculator.print_metrics(metrics)

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'split': args.split,
        'num_samples': len(dataset),
        'metrics': metrics,
        'predictions': results['predictions'][:100],  # Save first 100
        'ground_truths': results['ground_truths'][:100]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
