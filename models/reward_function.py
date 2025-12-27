"""Reward function for reinforcement learning"""
from typing import Any, Dict


class RewardFunction:
    """
    Reward function for RL-based template optimization

    Components:
    - R_acc: Accuracy reward
    - R_eff: Efficiency reward
    - R_int: Interpretability reward
    """

    def __init__(
        self,
        penalty_coef: float = 0.5,
        eff_weight: float = 0.3,
        int_weight: float = 0.2,
        max_steps: int = 15
    ):
        """
        Initialize reward function

        Args:
            penalty_coef: Penalty coefficient for wrong answers
            eff_weight: Weight for efficiency reward
            int_weight: Weight for interpretability reward
            max_steps: Maximum reasoning steps
        """
        self.alpha = penalty_coef
        self.beta = eff_weight
        self.gamma = int_weight
        self.max_steps = max_steps

    def compute_reward(
        self,
        reasoning_path: Any,
        ground_truth: str,
        step_index: int = None
    ) -> float:
        """
        Compute reward for a reasoning path

        Args:
            reasoning_path: ReasoningPath object
            ground_truth: Ground truth answer
            step_index: Current step index (for sequential rewards)

        Returns:
            Reward value
        """
        # Check answer correctness
        is_correct = self._check_answer(
            reasoning_path.final_answer,
            ground_truth
        )

        if is_correct:
            # Correct answer: combined reward
            r_acc = self._accuracy_reward(reasoning_path, ground_truth)
            r_eff = self._efficiency_reward(reasoning_path)
            r_int = self._interpretability_reward(reasoning_path)

            reward = r_acc + self.beta * r_eff + self.gamma * r_int
        else:
            # Wrong answer: "fail fast" penalty
            if step_index is not None:
                reward = -self.alpha * step_index
            else:
                reward = -self.alpha * len(reasoning_path.steps)

        return reward

    def _accuracy_reward(
        self,
        path: Any,
        ground_truth: str
    ) -> float:
        """
        Accuracy reward

        Args:
            path: ReasoningPath
            ground_truth: Ground truth answer

        Returns:
            Reward (1.0 or 0.0)
        """
        is_correct = self._check_answer(path.final_answer, ground_truth)
        return 1.0 if is_correct else 0.0

    def _efficiency_reward(self, path: Any) -> float:
        """
        Efficiency reward based on number of steps

        R_eff = 1 - (num_steps / max_steps)

        Args:
            path: ReasoningPath

        Returns:
            Efficiency reward [0, 1]
        """
        num_steps = len(path.steps)
        reward = max(0.0, 1.0 - num_steps / self.max_steps)
        return reward

    def _interpretability_reward(self, path: Any) -> float:
        """
        Interpretability reward based on step quality

        Args:
            path: ReasoningPath

        Returns:
            Interpretability reward [0, 1]
        """
        if len(path.steps) == 0:
            return 0.0

        # Compute average step length (in words)
        avg_length = sum(len(s.split()) for s in path.steps) / len(path.steps)

        # Optimal length (empirically determined)
        optimal_length = 10.0

        # Penalize steps that are too short or too long
        score = 1.0 - min(abs(avg_length - optimal_length) / optimal_length, 1.0)

        return score

    def _check_answer(
        self,
        prediction: str,
        ground_truth: str
    ) -> bool:
        """
        Check if prediction matches ground truth

        Supports:
        - Exact match
        - Normalized match (lowercase, stripped)
        - Numerical match (with tolerance)

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if match, False otherwise
        """
        # Normalize
        pred = str(prediction).strip().lower()
        gt = str(ground_truth).strip().lower()

        # Exact match
        if pred == gt:
            return True

        # Try numerical match
        try:
            pred_num = float(pred.replace(',', ''))
            gt_num = float(gt.replace(',', ''))

            # Allow small tolerance
            if abs(pred_num - gt_num) < 1e-3:
                return True
        except:
            pass

        # Check if one contains the other (for cases like "2020" vs "in 2020")
        if pred in gt or gt in pred:
            return True

        return False

    def compute_discounted_rewards(
        self,
        rewards: list,
        gamma: float = 0.95
    ) -> list:
        """
        Compute discounted cumulative rewards

        Args:
            rewards: List of step rewards
            gamma: Discount factor

        Returns:
            List of discounted rewards
        """
        discounted = []
        cumulative = 0

        # Compute backwards
        for r in reversed(rewards):
            cumulative = r + gamma * cumulative
            discounted.insert(0, cumulative)

        return discounted


class MultiObjectiveReward:
    """Multi-objective reward with weighted combination"""

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize multi-objective reward

        Args:
            weights: Dictionary of objective weights
        """
        self.weights = weights

    def compute(self, objectives: Dict[str, float]) -> float:
        """
        Compute weighted combination of objectives

        Args:
            objectives: Dictionary of objective values

        Returns:
            Combined reward
        """
        reward = 0.0
        for key, value in objectives.items():
            if key in self.weights:
                reward += self.weights[key] * value

        return reward
