"""Template management for dynamic prompt selection"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class TemplateManager:
    """Manage reasoning templates with RL-based selection"""

    def __init__(self, config: Any = None):
        """
        Initialize template manager

        Args:
            config: Configuration object
        """
        self.templates = self._initialize_templates()
        self.template_ids = list(self.templates.keys())

        # Policy network for template selection
        input_dim = 10  # Feature dimension
        self.policy_net = TemplatePolicyNetwork(
            input_dim=input_dim,
            num_templates=len(self.templates)
        )

        # Usage statistics
        self.usage_stats = defaultdict(lambda: {'success': 0, 'total': 0})

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize predefined templates"""
        templates = {
            'aggregation': """Let's solve this aggregation problem step by step:
1. Identify the column to aggregate
2. Apply any filter conditions
3. Apply the aggregation function (sum/count/avg/max/min)
4. Return the final result""",

            'comparison': """Let's solve this comparison problem step by step:
1. Extract the values to compare
2. Determine the comparison operator
3. Evaluate the comparison
4. Return the result""",

            'multi_hop': """Let's solve this multi-hop problem step by step:
1. First hop: Identify the initial information
2. Process the intermediate result
3. Second hop: Use the intermediate result
4. Return the final answer""",

            'arithmetic': """Let's solve this arithmetic problem step by step:
1. Identify the numbers to operate on
2. Determine the arithmetic operation
3. Perform the calculation
4. Return the result""",

            'filter': """Let's solve this filtering problem step by step:
1. Identify the filtering criteria
2. Apply the filter to select relevant rows
3. Extract the required information
4. Return the result""",

            'general': """Let's solve this step by step:
1. Understand what the question is asking
2. Locate the relevant information in the table
3. Process the information as needed
4. Return the final answer"""
        }

        return templates

    def select_template(
        self,
        table: Any,
        question: str,
        use_policy: bool = False
    ) -> Tuple[str, str]:
        """
        Select template for given table and question

        Args:
            table: Table object
            question: Question text
            use_policy: Whether to use policy network

        Returns:
            Tuple of (template_text, template_id)
        """
        if use_policy:
            # Use policy network
            features = self._extract_features(table, question)
            template_idx = self._select_with_policy(features)
            template_id = self.template_ids[template_idx]
        else:
            # Use heuristics
            template_id = self._select_with_heuristics(question)

        template = self.templates[template_id]
        return template, template_id

    def _select_with_heuristics(self, question: str) -> str:
        """Select template using keyword heuristics"""
        question_lower = question.lower()

        # Check for aggregation keywords
        agg_keywords = ['sum', 'total', 'count', 'how many', 'average', 'mean',
                       'maximum', 'minimum', 'most', 'least']
        if any(kw in question_lower for kw in agg_keywords):
            return 'aggregation'

        # Check for comparison keywords
        comp_keywords = ['more than', 'less than', 'greater', 'smaller',
                        'compare', 'difference', 'higher', 'lower']
        if any(kw in question_lower for kw in comp_keywords):
            return 'comparison'

        # Check for arithmetic keywords
        arith_keywords = ['+', '-', '×', '÷', 'add', 'subtract', 'multiply',
                         'divide', 'calculation']
        if any(kw in question_lower for kw in arith_keywords):
            return 'arithmetic'

        # Check for multi-hop indicators
        if ' and ' in question_lower or ' then ' in question_lower:
            return 'multi_hop'

        # Check for filtering
        filter_keywords = ['where', 'which', 'select', 'find', 'rows with']
        if any(kw in question_lower for kw in filter_keywords):
            return 'filter'

        # Default
        return 'general'

    def _select_with_policy(
        self,
        features: torch.Tensor
    ) -> int:
        """Select template using policy network"""
        with torch.no_grad():
            logits = self.policy_net(features)
            probs = F.softmax(logits, dim=-1)

            # Greedy selection (in training, would sample)
            template_idx = torch.argmax(probs).item()

        return template_idx

    def _extract_features(
        self,
        table: Any,
        question: str
    ) -> torch.Tensor:
        """
        Extract features for policy network

        Args:
            table: Table object
            question: Question text

        Returns:
            Feature tensor [feature_dim]
        """
        features = []

        # Table features
        num_rows = len(table.rows) / 100.0  # Normalize
        num_cols = len(table.headers) / 20.0
        features.extend([num_rows, num_cols])

        # Question features
        q_len = len(question.split()) / 50.0
        features.append(q_len)

        # Keyword features (binary)
        keywords_sets = [
            ['sum', 'total', 'count'],  # Aggregation
            ['compare', 'greater', 'less'],  # Comparison
            ['add', 'subtract'],  # Arithmetic
            ['where', 'which'],  # Filter
        ]

        question_lower = question.lower()
        for kw_set in keywords_sets:
            has_keyword = any(kw in question_lower for kw in kw_set)
            features.append(1.0 if has_keyword else 0.0)

        # Pad to fixed size (10)
        while len(features) < 10:
            features.append(0.0)

        return torch.tensor(features[:10], dtype=torch.float32)

    def update_stats(self, template_id: str, success: bool):
        """Update template usage statistics"""
        self.usage_stats[template_id]['total'] += 1
        if success:
            self.usage_stats[template_id]['success'] += 1

    def get_performance(self) -> Dict[str, float]:
        """Get success rate for each template"""
        performance = {}
        for tid, stats in self.usage_stats.items():
            if stats['total'] > 0:
                performance[tid] = stats['success'] / stats['total']
            else:
                performance[tid] = 0.0
        return performance


class TemplatePolicyNetwork(nn.Module):
    """Policy network for template selection"""

    def __init__(
        self,
        input_dim: int,
        num_templates: int,
        hidden_dim: int = 128
    ):
        """
        Initialize policy network

        Args:
            input_dim: Input feature dimension
            num_templates: Number of templates
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_templates)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            features: Input features [batch_size, input_dim] or [input_dim]

        Returns:
            Logits for template selection [batch_size, num_templates]
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        logits = self.network(features)
        return logits
