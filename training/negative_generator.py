"""Generate negative reasoning paths for contrastive learning"""
import json
import random
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.path_generator import ReasoningPath, ReasoningStep
from data.table_parser import TableParser


class NegativeGenerator:
    """
    Generate negative reasoning paths for contrastive learning

    Three types of negatives:
    1. Hard negatives: Structurally similar but semantically different
    2. Soft negatives: Partially correct paths with errors
    3. Adversarial negatives: Paths from other questions
    """

    def __init__(
        self,
        hard_negative_ratio: float = 0.4,
        soft_negative_ratio: float = 0.4,
        adversarial_negative_ratio: float = 0.2,
        max_negatives_per_sample: int = 5
    ):
        """
        Initialize negative generator

        Args:
            hard_negative_ratio: Ratio of hard negatives
            soft_negative_ratio: Ratio of soft negatives
            adversarial_negative_ratio: Ratio of adversarial negatives
            max_negatives_per_sample: Maximum negatives per sample
        """
        self.hard_ratio = hard_negative_ratio
        self.soft_ratio = soft_negative_ratio
        self.adversarial_ratio = adversarial_negative_ratio
        self.max_negatives = max_negatives_per_sample

        # Validate ratios sum to 1
        total_ratio = hard_negative_ratio + soft_negative_ratio + adversarial_negative_ratio
        assert abs(total_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    def generate_hard_negatives(
        self,
        positive_path: ReasoningPath,
        table: Any,
        num_negatives: int = 1
    ) -> List[ReasoningPath]:
        """
        Generate hard negatives: structurally similar but wrong

        Strategies:
        - Shuffle operation order
        - Change operation arguments
        - Use wrong columns/rows

        Args:
            positive_path: Positive reasoning path
            table: Table object
            num_negatives: Number of negatives to generate

        Returns:
            List of negative paths
        """
        negatives = []

        for _ in range(num_negatives):
            # Copy positive path
            neg_path = copy.deepcopy(positive_path)

            # Strategy 1: Shuffle steps (keeping structure)
            if len(neg_path.steps) > 2 and random.random() < 0.5:
                # Shuffle middle steps
                middle_steps = neg_path.steps[1:-1]
                random.shuffle(middle_steps)
                neg_path.steps = [neg_path.steps[0]] + middle_steps + [neg_path.steps[-1]]

            # Strategy 2: Change arguments
            if random.random() < 0.5:
                step_idx = random.randint(0, len(neg_path.steps) - 1)
                step = neg_path.steps[step_idx]

                # Change column/row if present
                if step.arguments:
                    arg_idx = random.randint(0, len(step.arguments) - 1)

                    # If argument is a column name, change to different column
                    if isinstance(step.arguments[arg_idx], str):
                        if hasattr(table, 'headers'):
                            other_cols = [h for h in table.headers if h != step.arguments[arg_idx]]
                            if other_cols:
                                step.arguments[arg_idx] = random.choice(other_cols)

            # Strategy 3: Insert wrong operation
            if random.random() < 0.3:
                wrong_ops = ['filter', 'select', 'sort', 'aggregate']
                wrong_step = ReasoningStep(
                    operation=random.choice(wrong_ops),
                    arguments=[],
                    result=""
                )
                insert_pos = random.randint(0, len(neg_path.steps))
                neg_path.steps.insert(insert_pos, wrong_step)

            negatives.append(neg_path)

        return negatives

    def generate_soft_negatives(
        self,
        positive_path: ReasoningPath,
        table: Any,
        num_negatives: int = 1
    ) -> List[ReasoningPath]:
        """
        Generate soft negatives: partially correct with errors

        Strategies:
        - Remove a critical step
        - Add unnecessary steps
        - Change final computation

        Args:
            positive_path: Positive reasoning path
            table: Table object
            num_negatives: Number of negatives to generate

        Returns:
            List of negative paths
        """
        negatives = []

        for _ in range(num_negatives):
            neg_path = copy.deepcopy(positive_path)

            # Strategy 1: Remove a step
            if len(neg_path.steps) > 2 and random.random() < 0.5:
                # Remove a random step (not first or last)
                remove_idx = random.randint(1, len(neg_path.steps) - 2)
                neg_path.steps.pop(remove_idx)

            # Strategy 2: Add unnecessary step
            if random.random() < 0.5:
                redundant_ops = ['identity', 'copy', 'duplicate']
                redundant_step = ReasoningStep(
                    operation=random.choice(redundant_ops),
                    arguments=[],
                    result=""
                )
                insert_pos = random.randint(0, len(neg_path.steps))
                neg_path.steps.insert(insert_pos, redundant_step)

            # Strategy 3: Change final operation
            if random.random() < 0.5 and neg_path.steps:
                last_step = neg_path.steps[-1]

                # Change aggregation function
                if last_step.operation in ['sum', 'mean', 'max', 'min']:
                    other_aggs = ['sum', 'mean', 'max', 'min', 'count']
                    other_aggs.remove(last_step.operation)
                    last_step.operation = random.choice(other_aggs)

            negatives.append(neg_path)

        return negatives

    def generate_adversarial_negatives(
        self,
        all_samples: List[Dict[str, Any]],
        current_idx: int,
        num_negatives: int = 1
    ) -> List[ReasoningPath]:
        """
        Generate adversarial negatives: paths from other questions

        Args:
            all_samples: All training samples
            current_idx: Index of current sample
            num_negatives: Number of negatives to generate

        Returns:
            List of negative paths
        """
        negatives = []

        # Get other samples
        other_indices = [i for i in range(len(all_samples)) if i != current_idx]

        for _ in range(num_negatives):
            if not other_indices:
                break

            # Sample from other questions
            other_idx = random.choice(other_indices)
            other_sample = all_samples[other_idx]

            # Use their positive path as our negative
            if 'reasoning_path' in other_sample:
                neg_path = ReasoningPath.from_dict(other_sample['reasoning_path'])
                negatives.append(neg_path)
            elif 'positive_path' in other_sample:
                neg_path = ReasoningPath.from_dict(other_sample['positive_path'])
                negatives.append(neg_path)

        return negatives

    def generate_negatives(
        self,
        positive_path: ReasoningPath,
        table: Any,
        all_samples: Optional[List[Dict[str, Any]]] = None,
        current_idx: Optional[int] = None
    ) -> List[ReasoningPath]:
        """
        Generate all types of negatives

        Args:
            positive_path: Positive reasoning path
            table: Table object
            all_samples: All samples (for adversarial negatives)
            current_idx: Current sample index

        Returns:
            List of negative paths
        """
        num_hard = int(self.max_negatives * self.hard_ratio)
        num_soft = int(self.max_negatives * self.soft_ratio)
        num_adversarial = self.max_negatives - num_hard - num_soft

        negatives = []

        # Generate hard negatives
        if num_hard > 0:
            negatives.extend(
                self.generate_hard_negatives(positive_path, table, num_hard)
            )

        # Generate soft negatives
        if num_soft > 0:
            negatives.extend(
                self.generate_soft_negatives(positive_path, table, num_soft)
            )

        # Generate adversarial negatives
        if num_adversarial > 0 and all_samples and current_idx is not None:
            negatives.extend(
                self.generate_adversarial_negatives(
                    all_samples, current_idx, num_adversarial
                )
            )

        return negatives

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = True
    ):
        """
        Process entire dataset to add negatives

        Args:
            input_path: Input JSON file path
            output_path: Output JSON file path
            verbose: Whether to show progress
        """
        # Load data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            elif 'examples' in data:
                data = data['examples']

        parser = TableParser()
        output_data = []

        iterator = tqdm(enumerate(data), total=len(data), desc="Generating negatives") if verbose else enumerate(data)

        for idx, sample in iterator:
            # Parse table
            table = parser.parse(sample['table'])

            # Get positive path
            if 'reasoning_path' in sample:
                positive_path = ReasoningPath.from_dict(sample['reasoning_path'])
            else:
                # If no reasoning path, create a simple one
                positive_path = ReasoningPath(steps=[
                    ReasoningStep(operation="answer", arguments=[], result=sample.get('answer', ''))
                ])

            # Generate negatives
            negative_paths = self.generate_negatives(
                positive_path=positive_path,
                table=table,
                all_samples=data,
                current_idx=idx
            )

            # Create output sample
            output_sample = {
                'table': sample['table'],
                'question': sample['question'],
                'answer': sample.get('answer', sample.get('label', '')),
                'positive_path': positive_path.to_dict(),
                'negative_paths': [neg.to_dict() for neg in negative_paths]
            }

            if 'table_id' in sample:
                output_sample['table_id'] = sample['table_id']

            output_data.append(output_sample)

        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"Processed {len(output_data)} samples")
            print(f"Saved to {output_path}")


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate negative samples')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--hard_ratio', type=float, default=0.4,
                       help='Hard negative ratio')
    parser.add_argument('--soft_ratio', type=float, default=0.4,
                       help='Soft negative ratio')
    parser.add_argument('--adversarial_ratio', type=float, default=0.2,
                       help='Adversarial negative ratio')
    parser.add_argument('--max_negatives', type=int, default=5,
                       help='Maximum negatives per sample')

    args = parser.parse_args()

    # Create generator
    generator = NegativeGenerator(
        hard_negative_ratio=args.hard_ratio,
        soft_negative_ratio=args.soft_ratio,
        adversarial_negative_ratio=args.adversarial_ratio,
        max_negatives_per_sample=args.max_negatives
    )

    # Process dataset
    generator.process_dataset(
        input_path=args.input,
        output_path=args.output,
        verbose=True
    )


if __name__ == '__main__':
    main()
