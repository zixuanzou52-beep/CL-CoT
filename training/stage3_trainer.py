"""Stage 3: Reinforcement learning fine-tuning trainer"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_trainer import BaseTrainer
from utils.config import Config
from utils.logger import ExperimentLogger
from models.reward_function import RewardFunction
from models.template_manager import TemplateManager
from data.path_generator import ReasoningPath


class PPOBuffer:
    """Buffer for storing PPO trajectories"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []

    def add(
        self,
        state: Any,
        action: Any,
        reward: float,
        value: float,
        log_prob: float
    ):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_advantages(self, gamma: float = 0.95, gae_lambda: float = 0.95):
        """Compute GAE advantages"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae

            advantages[t] = gae
            returns[t] = gae + values[t]

            next_value = values[t]

        self.advantages = advantages.tolist()
        self.returns = returns.tolist()

    def get_batch(self) -> Dict[str, List]:
        """Get all data as batch"""
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'old_log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns
        }

    def clear(self):
        """Clear buffer"""
        self.__init__()


class Stage3Trainer(BaseTrainer):
    """Trainer for Stage 3: RL fine-tuning with PPO"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        logger: Optional[ExperimentLogger] = None,
        checkpoint_manager: Optional[Any] = None,
        device: str = "cuda"
    ):
        """
        Initialize Stage 3 trainer

        Args:
            model: CL-CoT model
            config: Configuration
            train_loader: Training dataloader
            eval_loader: Evaluation dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            logger: Logger
            checkpoint_manager: Checkpoint manager
            device: Device
        """
        super().__init__(
            model=model,
            config=config,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            device=device
        )

        # Initialize reward function
        self.reward_function = RewardFunction(
            penalty_coef=config.get('reward.penalty_coef', 0.5),
            eff_weight=config.get('reward.eff_weight', 0.3),
            int_weight=config.get('reward.int_weight', 0.2),
            max_steps=config.get('reward.max_steps', 15)
        )

        # Initialize template manager
        self.template_manager = TemplateManager()

        # PPO parameters
        self.ppo_epsilon = config.get('rl.ppo_epsilon', 0.2)
        self.value_coef = config.get('rl.value_coef', 0.5)
        self.entropy_coef = config.get('rl.entropy_coef', 0.01)
        self.gamma = config.get('rl.gamma', 0.95)
        self.gae_lambda = config.get('rl.gae_lambda', 0.95)
        self.ppo_epochs = config.get('rl.ppo_epochs', 4)

        # Value network (simple MLP on top of model)
        self.value_head = nn.Sequential(
            nn.Linear(config.get('encoder.hidden_dim', 768), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        # Add value head to optimizer
        if self.optimizer:
            self.optimizer.add_param_group({
                'params': self.value_head.parameters(),
                'lr': config.get('training.stage3_lr')
            })

    def select_template(
        self,
        state: Dict[str, Any],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select template using policy

        Args:
            state: Current state (table, question, partial path)
            deterministic: Whether to use deterministic selection

        Returns:
            (template_id, log_prob, value)
        """
        # Encode state
        table = state['table']
        question = state['question']

        # Create input text
        from data.table_parser import TableParser
        parser = TableParser()
        table_text = parser.linearize(table, format="markdown")
        input_text = f"Table:\n{table_text}\n\nQuestion: {question}\n\nNext template:"

        # Tokenize
        inputs = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get model hidden states
        with torch.no_grad():
            outputs = self.model.base_model(
                **inputs,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # [1, L, d]
            state_emb = hidden_states.mean(dim=1)  # [1, d]

        # Get value
        value = self.value_head(state_emb).squeeze(-1).squeeze(0)  # []

        # Get template distribution (simple: uniform over available templates)
        num_templates = len(self.template_manager.templates)
        template_probs = torch.ones(num_templates) / num_templates
        template_probs = template_probs.to(self.device)

        # Sample template
        if deterministic:
            template_id = template_probs.argmax().item()
        else:
            dist = torch.distributions.Categorical(template_probs)
            template_id = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(template_id).to(self.device))

        if deterministic:
            return template_id, torch.tensor(0.0, device=self.device), value
        return template_id, log_prob, value

    def collect_episode(
        self,
        table: Any,
        question: str,
        answer: str,
        max_steps: int = 15
    ) -> PPOBuffer:
        """
        Collect one episode of experience

        Args:
            table: Table object
            question: Question string
            answer: Ground truth answer
            max_steps: Maximum reasoning steps

        Returns:
            PPOBuffer with episode data
        """
        buffer = PPOBuffer()
        current_path = ReasoningPath(steps=[])

        state = {
            'table': table,
            'question': question,
            'partial_path': current_path
        }

        for step_idx in range(max_steps):
            # Select template
            template_id, log_prob, value = self.select_template(state)

            # Execute template (simplified - in practice would use actual execution)
            template = self.template_manager.get_template(template_id)

            # Mock step execution
            from data.path_generator import ReasoningStep
            step = ReasoningStep(
                operation=template.name,
                arguments=[],
                result=""
            )
            current_path.steps.append(step)

            # Compute reward
            reward = self.reward_function.compute_reward(
                path=current_path,
                ground_truth=answer,
                table=table
            )

            # Add to buffer
            buffer.add(
                state=state.copy(),
                action=template_id,
                reward=reward,
                value=value.detach().item(),
                log_prob=log_prob.detach().item()
            )

            # Check if done (simplified)
            if step_idx >= max_steps - 1:
                break

            # Update state
            state['partial_path'] = current_path

        return buffer

    def update_policy(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        Update policy using PPO

        Args:
            buffer: Experience buffer

        Returns:
            Metrics dictionary
        """
        # Compute advantages
        buffer.compute_advantages(self.gamma, self.gae_lambda)

        batch = buffer.get_batch()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # PPO epochs
        for _ in range(self.ppo_epochs):
            for i in range(len(batch['states'])):
                state = batch['states'][i]
                _action = batch['actions'][i]
                old_log_prob = batch['old_log_probs'][i]
                advantage = batch['advantages'][i]
                return_val = batch['returns'][i]

                # Get current policy
                _, new_log_prob, value = self.select_template(state)

                # PPO clipped objective
                ratio = torch.exp(
                    new_log_prob - torch.tensor(old_log_prob, device=self.device)
                )
                advantage_t = torch.tensor(advantage).to(self.device)

                surr1 = ratio * advantage_t
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_epsilon,
                    1.0 + self.ppo_epsilon
                ) * advantage_t

                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                return_t = torch.tensor(return_val).to(self.device)
                value_loss = (value - return_t).pow(2)

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        num_updates = len(batch['states']) * self.ppo_epochs

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'avg_reward': float(np.mean(batch['rewards']))
        }

    def train_epoch(self) -> float:
        """
        Train for one epoch with RL

        Returns:
            Average reward
        """
        self.model.train()
        epoch_reward = 0
        num_episodes = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1} - RL",
            disable=not self.logger
        )

        for batch_idx, batch in enumerate(progress_bar):
            tables = batch['tables']
            questions = batch['questions']
            answers = batch['answers']

            batch_size = len(questions)

            for i in range(batch_size):
                # Collect episode
                buffer = self.collect_episode(
                    table=tables[i],
                    question=questions[i],
                    answer=answers[i]
                )

                # Update policy
                metrics = self.update_policy(buffer)

                # Track metrics
                epoch_reward += metrics['avg_reward']
                num_episodes += 1

                self.log_metrics(metrics)

            # Update progress bar
            avg_reward = epoch_reward / max(num_episodes, 1)
            progress_bar.set_postfix({'avg_reward': f"{avg_reward:.4f}"})

        return epoch_reward / max(num_episodes, 1)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate RL policy

        Returns:
            Evaluation metrics
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_reward = 0
        total_episodes = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                tables = batch['tables']
                questions = batch['questions']
                answers = batch['answers']

                for i in range(len(questions)):
                    buffer = self.collect_episode(
                        table=tables[i],
                        question=questions[i],
                        answer=answers[i]
                    )

                    total_reward += np.mean(buffer.rewards)
                    total_episodes += 1

        avg_reward = total_reward / max(total_episodes, 1)

        return {
            'reward': avg_reward
        }
