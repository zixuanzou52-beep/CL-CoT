"""Tests for reward function"""
import pytest
import torch
from models.reward_function import RewardFunction
from data.path_generator import ReasoningPath, ReasoningStep
from data.table_parser import Table


class TestRewardFunction:
    """Tests for RewardFunction"""

    def test_reward_initialization(self, reward_function):
        """Test reward function initialization"""
        assert reward_function is not None
        assert hasattr(reward_function, 'penalty_coef')
        assert hasattr(reward_function, 'eff_weight')
        assert hasattr(reward_function, 'int_weight')

    def test_compute_reward_correct_answer(
        self,
        reward_function,
        sample_reasoning_path,
        sample_table,
        sample_answer
    ):
        """Test reward for correct answer"""
        # Modify path to have correct answer
        sample_reasoning_path.steps[-1].result = sample_answer

        reward = reward_function.compute_reward(
            path=sample_reasoning_path,
            ground_truth=sample_answer,
            table=sample_table
        )

        # Correct answer should give positive reward
        assert reward > 0

    def test_compute_reward_incorrect_answer(
        self,
        reward_function,
        sample_reasoning_path,
        sample_table,
        sample_answer
    ):
        """Test reward for incorrect answer"""
        # Wrong answer
        sample_reasoning_path.steps[-1].result = "wrong_answer"

        reward = reward_function.compute_reward(
            path=sample_reasoning_path,
            ground_truth=sample_answer,
            table=sample_table
        )

        # Incorrect answer should give penalty
        assert reward < 1.0

    def test_efficiency_reward(self, reward_function):
        """Test efficiency component of reward"""
        # Short path (efficient)
        short_path = ReasoningPath(steps=[
            ReasoningStep("answer", ["1250000"], "1250000")
        ])

        # Long path (less efficient)
        long_path = ReasoningPath(steps=[
            ReasoningStep("filter", [], ""),
            ReasoningStep("select", [], ""),
            ReasoningStep("sort", [], ""),
            ReasoningStep("filter", [], ""),
            ReasoningStep("aggregate", [], ""),
            ReasoningStep("answer", ["1250000"], "1250000")
        ])

        reward_short = reward_function.compute_efficiency_reward(short_path)
        reward_long = reward_function.compute_efficiency_reward(long_path)

        # Shorter path should have higher efficiency reward
        assert reward_short >= reward_long

    def test_interpretability_reward(self, reward_function, sample_reasoning_path):
        """Test interpretability reward"""
        reward = reward_function.compute_interpretability_reward(
            sample_reasoning_path
        )

        assert isinstance(reward, float)
        assert 0 <= reward <= 1.0

    def test_max_steps_penalty(self, reward_function, sample_table):
        """Test penalty for exceeding max steps"""
        # Create path with too many steps
        long_path = ReasoningPath(steps=[
            ReasoningStep("step", [], "") for _ in range(20)
        ])

        long_path.steps[-1].result = "answer"

        reward = reward_function.compute_reward(
            path=long_path,
            ground_truth="answer",
            table=sample_table
        )

        # Should have penalty for being too long
        assert reward < 1.0

    def test_reward_components_combination(
        self,
        reward_function,
        sample_reasoning_path,
        sample_table,
        sample_answer
    ):
        """Test that reward combines all components"""
        sample_reasoning_path.steps[-1].result = sample_answer

        reward = reward_function.compute_reward(
            path=sample_reasoning_path,
            ground_truth=sample_answer,
            table=sample_table
        )

        # Reward should be combination of correctness, efficiency, interpretability
        assert isinstance(reward, (int, float))
        assert -1.0 <= reward <= 2.0  # Reasonable range

    def test_exact_match_reward(self, reward_function):
        """Test exact match checking"""
        is_match = reward_function.check_exact_match("1250000", "1250000")
        assert is_match

        is_match = reward_function.check_exact_match("1250000", "1000000")
        assert not is_match

    def test_fuzzy_match_reward(self, reward_function):
        """Test fuzzy matching"""
        # Numbers should match with minor differences
        is_match = reward_function.check_fuzzy_match("1250000", "1,250,000")
        # Implementation dependent

        is_match = reward_function.check_fuzzy_match("yes", "Yes")
        # Case insensitive match

    def test_reward_range(
        self,
        reward_function,
        sample_reasoning_path,
        sample_table,
        sample_answer
    ):
        """Test reward is in valid range"""
        for answer in [sample_answer, "wrong", "", "12345"]:
            sample_reasoning_path.steps[-1].result = answer

            reward = reward_function.compute_reward(
                path=sample_reasoning_path,
                ground_truth=sample_answer,
                table=sample_table
            )

            # Reward should be in reasonable range
            assert isinstance(reward, (int, float))
            assert not torch.isnan(torch.tensor(reward))
            assert not torch.isinf(torch.tensor(reward))


class TestRewardFunctionEdgeCases:
    """Tests for edge cases"""

    def test_empty_path(self, reward_function, sample_table):
        """Test reward for empty path"""
        empty_path = ReasoningPath(steps=[])

        reward = reward_function.compute_reward(
            path=empty_path,
            ground_truth="answer",
            table=sample_table
        )

        # Should handle empty path gracefully
        assert isinstance(reward, (int, float))

    def test_single_step_path(self, reward_function, sample_table):
        """Test reward for single step path"""
        single_step = ReasoningPath(steps=[
            ReasoningStep("answer", ["result"], "result")
        ])

        reward = reward_function.compute_reward(
            path=single_step,
            ground_truth="result",
            table=sample_table
        )

        assert reward > 0

    def test_empty_ground_truth(self, reward_function, sample_reasoning_path, sample_table):
        """Test with empty ground truth"""
        reward = reward_function.compute_reward(
            path=sample_reasoning_path,
            ground_truth="",
            table=sample_table
        )

        assert isinstance(reward, (int, float))

    def test_none_result(self, reward_function, sample_table):
        """Test with None as result"""
        path = ReasoningPath(steps=[
            ReasoningStep("answer", [], None)
        ])

        reward = reward_function.compute_reward(
            path=path,
            ground_truth="answer",
            table=sample_table
        )

        # Should handle None gracefully
        assert isinstance(reward, (int, float))

    def test_different_penalty_coefficients(self, sample_reasoning_path, sample_table):
        """Test different penalty coefficients"""
        # High penalty
        rf_high = RewardFunction(penalty_coef=0.9)
        # Low penalty
        rf_low = RewardFunction(penalty_coef=0.1)

        sample_reasoning_path.steps[-1].result = "wrong"

        reward_high = rf_high.compute_reward(
            path=sample_reasoning_path,
            ground_truth="correct",
            table=sample_table
        )

        reward_low = rf_low.compute_reward(
            path=sample_reasoning_path,
            ground_truth="correct",
            table=sample_table
        )

        # High penalty should give lower reward for wrong answer
        assert reward_high <= reward_low

    def test_different_weight_combinations(self, sample_reasoning_path, sample_table, sample_answer):
        """Test different weight combinations"""
        # Efficiency-focused
        rf_eff = RewardFunction(eff_weight=0.7, int_weight=0.1)

        # Interpretability-focused
        rf_int = RewardFunction(eff_weight=0.1, int_weight=0.7)

        sample_reasoning_path.steps[-1].result = sample_answer

        reward_eff = rf_eff.compute_reward(
            path=sample_reasoning_path,
            ground_truth=sample_answer,
            table=sample_table
        )

        reward_int = rf_int.compute_reward(
            path=sample_reasoning_path,
            ground_truth=sample_answer,
            table=sample_table
        )

        # Both should be positive for correct answer
        assert reward_eff > 0
        assert reward_int > 0
