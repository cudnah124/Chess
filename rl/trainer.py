"""
AlphaZero Trainer - RL training with self-play
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple

from rl.mcts.search import MCTS
from rl.self_play.game import generate_self_play_game
from rl.self_play.replay_buffer import ReplayBuffer


class AlphaZeroLoss(nn.Module):
    """Combined policy and value loss for AlphaZero"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        """
        Compute combined loss
        
        Args:
            policy_pred: Policy logits (batch, 4672)
            value_pred: Value predictions (batch, 1)
            policy_target: Target policy distribution (batch, 4672)
            value_target: Target values (batch, 1)
        
        Returns:
            total_loss, policy_loss, value_loss
        """
        # Policy loss: cross-entropy
        log_policy = torch.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_target * log_policy).sum(dim=1).mean()
        
        # Value loss: MSE
        value_loss = self.mse(value_pred, value_target)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss


class AlphaZeroTrainer:
    """
    AlphaZero trainer with self-play and training
    
    Args:
        model: Neural network model
        config: Configuration object
        device: PyTorch device
    """
    
    def __init__(self, model: nn.Module, config, device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.rl_lr,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5
        )
        
        # MCTS
        self.mcts = MCTS(self.model, config, device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=100000)
        
        # Loss function
        self.criterion = AlphaZeroLoss()
    
    def train_iteration(
        self,
        num_games: int,
        num_simulations: int,
        eval_vs_random: bool = False
    ) -> Tuple[int, int, int, float, int]:
        """
        Run one training iteration: self-play + training
        
        Args:
            num_games: Number of self-play games
            num_simulations: MCTS simulations per move
            eval_vs_random: Whether to evaluate vs random
        
        Returns:
            wins, losses, draws, winrate, buffer_size
        """
        # Self-play
        self.model.eval()
        
        for i in tqdm(range(num_games), desc="Self-play"):
            samples, outcome, _ = generate_self_play_game(
                self.mcts,
                num_simulations=num_simulations,
                temperature_threshold=30,
                max_moves=200
            )
            self.replay_buffer.add_batch(samples)
        
        # Training
        self._train_on_buffer(
            epochs=5,
            batch_size=self.config.rl_batch_size
        )
        
        # Evaluation (optional)
        wins, losses, draws, winrate = 0, 0, 0, 0.0
        if eval_vs_random:
            from core.utils.evaluation import evaluate_vs_random
            wins, losses, draws, winrate = evaluate_vs_random(
                self.model,
                self.config,
                num_games=self.config.eval_games
            )
            pass
        
        return wins, losses, draws, winrate, len(self.replay_buffer)
    
    def _train_on_buffer(self, epochs: int, batch_size: int):
        """Train on replay buffer"""
        self.model.train()
        
        batches_per_epoch = max(1, len(self.replay_buffer) // batch_size)
        total_steps = batches_per_epoch * epochs
        
        for step in range(total_steps):
            # Sample batch
            batch = self.replay_buffer.sample(batch_size)
            states, policies, values = zip(*batch)
            
            # Convert to tensors
            states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
            policies = torch.stack([torch.FloatTensor(p) for p in policies]).to(self.device)
            values = torch.FloatTensor(values).unsqueeze(1).to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(states)
            
            # Compute loss
            loss, p_loss, v_loss = self.criterion(policy_pred, value_pred, policies, values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Logging
            if step % 100 == 0:
                epoch = step // batches_per_epoch + 1
                print(f"  Epoch {epoch}/{epochs}, Step {step}/{total_steps}: "
                      f"Loss={loss.item():.4f}, P={p_loss.item():.4f}, V={v_loss.item():.4f}")
        
        # Step scheduler
        self.scheduler.step()
