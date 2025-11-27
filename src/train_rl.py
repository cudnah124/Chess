import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import os
from collections import deque
import random
from tqdm import tqdm

try:
    from config import *
    from model import SmallResNet
    from mcts import MCTS
    from parse_game import board_to_tensor
except ImportError as e:
    print(f"Warning: {e}")

#Can change

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, policy_target, value_target):
        self.buffer.append((state, policy_target, value_target))
    
    def add_batch(self, samples):
        self.buffer.extend(samples)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class SelfPlayDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state, policy, value = self.samples[idx]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
            torch.FloatTensor([value])
        )


class AlphaZeroTrainer:
    def __init__(self, model, device='cuda', lr=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.replay_buffer = ReplayBuffer(max_size=100000)
        
    def combined_loss(self, policy_pred, value_pred, policy_target, value_target, lambda_val=0.01):
        policy_loss = -(policy_target * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
        value_loss = nn.MSELoss()(value_pred, value_target)
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss, value_loss
    
    def self_play_game(self, mcts_simulations=800, temperature=1.0):
        board = chess.Board()
        samples = []
        game_history = []
        
        mcts = MCTS(self.model, device=self.device)
        
        while not board.is_game_over():
            state = board_to_tensor(board, game_history)
            policy = mcts.search(board, num_simulations=mcts_simulations, temperature=temperature)
            
            samples.append((state, policy, None))
            game_history.append((state, policy))
            
            legal_moves = list(board.legal_moves)
            move_probs = [(move, policy[self.move_to_index(move)]) for move in legal_moves]
            
            moves, probs = zip(*move_probs)
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            selected_move = np.random.choice(moves, p=probs)
            board.push(selected_move)
        
        result = board.result()
        if result == "1-0":
            game_outcome = 1.0
        elif result == "0-1":
            game_outcome = -1.0
        else:
            game_outcome = 0.0
        
        for i, (state, policy, _) in enumerate(samples):
            if i % 2 == 0:
                value = game_outcome
            else:
                value = -game_outcome
            samples[i] = (state, policy, value)
        
        return samples
    
    def move_to_index(self, move):
        from_sq = move.from_square
        to_sq = move.to_square
        return from_sq * 64 + to_sq
    
    def train_on_batch(self, batch):
        """
        Train model trên một batch từ Replay Buffer
        """
        self.model.train()
        
        states, policies, values = zip(*batch)
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        policies = torch.stack([torch.FloatTensor(p) for p in policies]).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        # Forward pass
        policy_pred, value_pred = self.model(states)
        
        # Compute loss
        loss, policy_loss, value_loss = self.combined_loss(
            policy_pred, value_pred, policies, values
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def train_epoch(self, num_games=100, mcts_simulations=800, batch_size=64, epochs_per_iteration=10):
        """
        Một iteration của AlphaZero training:
        1. Self-Play để sinh dữ liệu
        2. Train trên dữ liệu đó
        """
        print(f"\n=== Self-Play: Generating {num_games} games ===")
        
        # Phase 1: Self-Play
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_games)):
                samples = self.self_play_game(mcts_simulations=mcts_simulations)
                self.replay_buffer.add_batch(samples)
        
        print(f"Replay Buffer size: {len(self.replay_buffer)}")
        
        # Phase 2: Training
        print(f"\n=== Training on Self-Play Data ===")
        for epoch in range(epochs_per_iteration):
            # Sample từ replay buffer
            batch = self.replay_buffer.sample(batch_size)
            
            loss, policy_loss, value_loss = self.train_on_batch(batch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}")
    
    def train(self, num_iterations=100, games_per_iteration=100, save_interval=10):
        """
        Main training loop cho AlphaZero RL
        """
        print("="*60)
        print("Starting AlphaZero RL Training Loop")
        print("="*60)
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # Train một iteration
            self.train_epoch(num_games=games_per_iteration)
            
            # Save checkpoint
            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(f"models/model_rl_iter_{iteration+1}.pth")
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
    
    def save_checkpoint(self, path):
        """Lưu model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_size': len(self.replay_buffer)
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")


def main():
    """
    Main function để chạy AlphaZero RL Training
    """
    # Hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model từ Phase 1 (Supervised Learning)
    model = SmallResNet(
        input_channels=32,
        num_res_blocks=5,
        action_space_size=4672
    )
    
    # Load pretrained weights từ supervised training
    supervised_model_path = "models/model_supervised.pth"
    if os.path.exists(supervised_model_path):
        print(f"Loading supervised model from {supervised_model_path}")
        model.load_state_dict(torch.load(supervised_model_path, map_location=device))
    else:
        print("Warning: No supervised model found. Starting from scratch.")
    
    # Khởi tạo Trainer
    trainer = AlphaZeroTrainer(
        model=model,
        device=device,
        lr=0.001,
        weight_decay=1e-4
    )
    
    # Bắt đầu training
    trainer.train(
        num_iterations=100,
        games_per_iteration=50,  # Số games self-play mỗi iteration
        save_interval=5
    )
    
    # Save final model
    trainer.save_checkpoint("models/model_rl_final.pth")
    print("\nTraining complete! Model saved to models/model_rl_final.pth")


if __name__ == '__main__':
    main()