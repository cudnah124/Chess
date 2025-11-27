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
        self.model.train()
        
        states, policies, values = zip(*batch)
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        policies = torch.stack([torch.FloatTensor(p) for p in policies]).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        policy_pred, value_pred = self.model(states)
        loss, policy_loss, value_loss = self.combined_loss(
            policy_pred, value_pred, policies, values
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def train_epoch(self, num_games=100, mcts_simulations=800, batch_size=64, epochs_per_iteration=10):
        print(f"\n=== Self-Play: {num_games} games ===")
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_games)):
                samples = self.self_play_game(mcts_simulations=mcts_simulations)
                self.replay_buffer.add_batch(samples)
        
        print(f"Buffer size: {len(self.replay_buffer)}")
        
        print(f"\n=== Training ===")
        for epoch in range(epochs_per_iteration):
            batch = self.replay_buffer.sample(batch_size)
            loss, policy_loss, value_loss = self.train_on_batch(batch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, P={policy_loss:.4f}, V={value_loss:.4f}")
    
    def train(self, num_iterations=100, games_per_iteration=100, save_interval=10):
        print("="*60)
        print("AlphaZero RL Training")
        print("="*60)
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            self.train_epoch(num_games=games_per_iteration)
            
            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(f"models/model_rl_iter_{iteration+1}.pth")
        
        print("\nTraining completed!")
    
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_size': len(self.replay_buffer)
        }, path)
        print(f"Saved: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded: {path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = SmallResNet(
        input_channels=32,
        num_res_blocks=5,
        action_space_size=4672
    )
    
    supervised_path = "models/model_supervised.pth"
    if os.path.exists(supervised_path):
        print(f"Loading: {supervised_path}")
        model.load_state_dict(torch.load(supervised_path, map_location=device))
    else:
        print("Warning: No supervised model found")
    
    trainer = AlphaZeroTrainer(
        model=model,
        device=device,
        lr=0.001,
        weight_decay=1e-4
    )
    
    trainer.train(
        num_iterations=100,
        games_per_iteration=50,
        save_interval=5
    )
    
    trainer.save_checkpoint("models/model_rl_final.pth")
    print("\nDone!")


if __name__ == '__main__':
    main()