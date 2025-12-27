"""Reasoning path generation and management"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import torch
from enum import Enum


class OperationType(Enum):
    """Types of reasoning operations"""
    IDENTIFY = "identify"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    SORT = "sort"
    ARITHMETIC = "arithmetic"
    COMPARE = "compare"
    EXTRACT = "extract"
    RETURN = "return"
    UNKNOWN = "unknown"


@dataclass
class ReasoningPath:
    """Reasoning path data structure"""
    steps: List[str]
    operations: List[str] = field(default_factory=list)
    intermediate_results: List[Any] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Get number of steps"""
        return len(self.steps)

    def __getitem__(self, idx: int) -> str:
        """Get step by index"""
        return self.steps[idx]

    def add_step(
        self,
        step: str,
        operation: str = "unknown",
        result: Any = None
    ):
        """
        Add a step to the reasoning path

        Args:
            step: Step description
            operation: Operation type
            result: Intermediate result
        """
        self.steps.append(step)
        self.operations.append(operation)
        self.intermediate_results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'steps': self.steps,
            'operations': self.operations,
            'intermediate_results': self.intermediate_results,
            'final_answer': self.final_answer,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningPath':
        """Create from dictionary"""
        return cls(
            steps=data.get('steps', []),
            operations=data.get('operations', []),
            intermediate_results=data.get('intermediate_results', []),
            final_answer=data.get('final_answer', ''),
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {})
        )

    def to_text(self) -> str:
        """Convert to text representation"""
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step}")
        if self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


class ReasoningPathGenerator:
    """Generate reasoning paths using language model"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        template_manager: Any = None,
        num_paths: int = 5,
        temperature: float = 0.8,
        max_steps: int = 15,
        device: str = "cuda"
    ):
        """
        Initialize path generator

        Args:
            model: Language model
            tokenizer: Tokenizer
            template_manager: Template manager for prompts
            num_paths: Number of paths to generate
            temperature: Sampling temperature
            max_steps: Maximum reasoning steps
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.template_manager = template_manager
        self.num_paths = num_paths
        self.temperature = temperature
        self.max_steps = max_steps
        self.device = device

    def generate(
        self,
        table: Any,
        question: str,
        n: Optional[int] = None
    ) -> List[ReasoningPath]:
        """
        Generate multiple reasoning paths

        Args:
            table: Table object
            question: Question text
            n: Number of paths (uses self.num_paths if None)

        Returns:
            List of ReasoningPath objects
        """
        if n is None:
            n = self.num_paths

        paths = []

        # Get template if available
        template = None
        if self.template_manager:
            template, _ = self.template_manager.select_template(table, question)

        # Generate n paths
        for i in range(n):
            path = self._generate_single_path(table, question, template)
            paths.append(path)

        return paths

    def _generate_single_path(
        self,
        table: Any,
        question: str,
        template: Optional[str] = None
    ) -> ReasoningPath:
        """
        Generate a single reasoning path

        Args:
            table: Table object
            question: Question text
            template: Optional template

        Returns:
            ReasoningPath object
        """
        # Build prompt
        prompt = self._build_prompt(table, question, template)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Parse into ReasoningPath
        path = self._parse_generated_text(generated_text)

        return path

    def _build_prompt(
        self,
        table: Any,
        question: str,
        template: Optional[str] = None
    ) -> str:
        """
        Build prompt for generation

        Args:
            table: Table object
            question: Question text
            template: Optional template

        Returns:
            Prompt string
        """
        # Import here to avoid circular dependency
        from .table_parser import TableParser

        parser = TableParser()
        table_text = parser.linearize(table, format="markdown")

        if template:
            prompt = f"""Table:
{table_text}

Question: {question}

{template}

Let's solve this step by step:
"""
        else:
            prompt = f"""Table:
{table_text}

Question: {question}

Let's solve this step by step:
1."""

        return prompt

    def _parse_generated_text(self, text: str) -> ReasoningPath:
        """
        Parse generated text into ReasoningPath

        Args:
            text: Generated text

        Returns:
            ReasoningPath object
        """
        steps = []
        operations = []
        final_answer = ""

        # Split into lines
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Check if it's a step
            if line.startswith(('Step', '1.', '2.', '3.', '4.', '5.')):
                # Clean up step text
                step_text = line
                for prefix in ['Step 1:', 'Step 2:', 'Step 3:', 'Step 4:', 'Step 5:',
                              '1.', '2.', '3.', '4.', '5.']:
                    if step_text.startswith(prefix):
                        step_text = step_text[len(prefix):].strip()
                        break

                steps.append(step_text)

                # Identify operation type
                operation = self._identify_operation(step_text)
                operations.append(operation)

            # Check for final answer
            elif line.lower().startswith(('answer:', 'final answer:', 'result:')):
                final_answer = line.split(':', 1)[1].strip() if ':' in line else line

        # If no steps found, treat whole text as one step
        if not steps and text.strip():
            steps = [text.strip()]
            operations = [self._identify_operation(text.strip())]

        path = ReasoningPath(
            steps=steps,
            operations=operations,
            final_answer=final_answer
        )

        return path

    def _identify_operation(self, step_text: str) -> str:
        """
        Identify operation type from step text

        Args:
            step_text: Step description

        Returns:
            Operation type string
        """
        step_lower = step_text.lower()

        # Define keywords for each operation
        operation_keywords = {
            'filter': ['filter', 'where', 'select rows', 'rows where'],
            'aggregate': ['sum', 'count', 'average', 'avg', 'max', 'min', 'total'],
            'sort': ['sort', 'order', 'rank', 'arrange'],
            'arithmetic': ['+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide'],
            'compare': ['compare', 'greater', 'less', 'more than', 'less than'],
            'identify': ['identify', 'find column', 'locate'],
            'extract': ['extract', 'get value', 'retrieve'],
            'return': ['return', 'output', 'result']
        }

        # Check for keywords
        for op_type, keywords in operation_keywords.items():
            if any(kw in step_lower for kw in keywords):
                return op_type

        return 'unknown'

    def generate_with_template(
        self,
        table: Any,
        question: str,
        template: str
    ) -> ReasoningPath:
        """
        Generate path with specific template

        Args:
            table: Table object
            question: Question text
            template: Template string

        Returns:
            ReasoningPath object
        """
        return self._generate_single_path(table, question, template)
