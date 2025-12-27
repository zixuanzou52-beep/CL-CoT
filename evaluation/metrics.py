"""Evaluation metrics for table QA"""
import numpy as np
from typing import List, Dict, Any
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text"""
    text = str(text).strip().lower()
    # Remove punctuation
    text = text.replace(',', '').replace('.', '')
    return text


def compute_exact_match(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute exact match score

    Args:
        predictions: List of predictions
        ground_truths: List of ground truths

    Returns:
        Exact match score [0, 1]
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        if pred_norm == gt_norm:
            correct += 1

    return correct / len(predictions) if len(predictions) > 0 else 0.0


def compute_f1(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute F1 score (token-level)

    Args:
        predictions: List of predictions
        ground_truths: List of ground truths

    Returns:
        Average F1 score
    """
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = normalize_answer(pred).split()
        gt_tokens = normalize_answer(gt).split()

        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            f1 = 1.0
        elif len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = 0.0
        else:
            # Compute token overlap
            pred_counter = Counter(pred_tokens)
            gt_counter = Counter(gt_tokens)

            common_tokens = sum((pred_counter & gt_counter).values())

            precision = common_tokens / len(pred_tokens)
            recall = common_tokens / len(gt_tokens)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

        f1_scores.append(f1)

    return np.mean(f1_scores) if f1_scores else 0.0


class MetricsCalculator:
    """Calculate various evaluation metrics"""

    def __init__(self):
        pass

    def exact_match(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute exact match"""
        return compute_exact_match(predictions, ground_truths)

    def f1_score(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute F1 score"""
        return compute_f1(predictions, ground_truths)

    def accuracy(self, predictions: List[Any], ground_truths: List[Any]) -> float:
        """Compute accuracy"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Length mismatch")

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        return correct / len(predictions) if len(predictions) > 0 else 0.0

    def compute_all(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Compute all metrics

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'exact_match': self.exact_match(predictions, ground_truths),
            'f1_score': self.f1_score(predictions, ground_truths),
            'num_samples': len(predictions)
        }

        return metrics

    def efficiency_metrics(
        self,
        reasoning_paths: List[Any],
        inference_times: List[float]
    ) -> Dict[str, float]:
        """
        Compute efficiency metrics

        Args:
            reasoning_paths: List of ReasoningPath objects
            inference_times: List of inference times (seconds)

        Returns:
            Dictionary of efficiency metrics
        """
        if not reasoning_paths:
            return {}

        avg_steps = np.mean([len(p.steps) for p in reasoning_paths])
        avg_time = np.mean(inference_times) if inference_times else 0.0

        # Token efficiency
        total_tokens = sum(
            sum(len(step.split()) for step in p.steps)
            for p in reasoning_paths
        )
        token_efficiency = len(reasoning_paths) / total_tokens if total_tokens > 0 else 0.0

        return {
            'avg_steps': float(avg_steps),
            'avg_time': float(avg_time),
            'token_efficiency': float(token_efficiency)
        }

    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in formatted way"""
        print("\n" + "="*50)
        print("Evaluation Metrics")
        print("="*50)

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")

        print("="*50 + "\n")
