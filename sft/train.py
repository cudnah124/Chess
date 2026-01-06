"""
SFT Training Script - Supervised Fine-Tuning from PGN games
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

from core.config import Config
from core.models.resnet import SmallResNet
from core.utils.checkpoint import save_checkpoint
from sft.data.dataset import ChessPGNDataset
from sft.loss import AlphaZeroLoss


def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch in pbar:
        states = batch['state'].to(device)
        policy_targets = batch['policy'].to(device)
        value_targets = batch['value'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        policy_pred, value_pred = model(states)
        
        # Compute loss
        loss, p_loss, v_loss = criterion(policy_pred, value_pred, policy_targets, value_targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_policy_loss += p_loss.item()
        total_value_loss += v_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'p_loss': f'{p_loss.item():.4f}',
            'v_loss': f'{v_loss.item():.4f}'
        })
    
    # Average losses
    avg_loss = total_loss / len(loader)
    avg_p_loss = total_policy_loss / len(loader)
    avg_v_loss = total_value_loss / len(loader)
    
    return avg_loss, avg_p_loss, avg_v_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/sft_config.yaml'
    
    if not os.path.exists(config_path):
        config = Config()
        config.to_yaml(config_path)
    
    config = Config.from_yaml(config_path)
    config.validate()
    
    model = SmallResNet(config).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.sft_lr,
        weight_decay=config.weight_decay
    )
    criterion = AlphaZeroLoss()
    
    dataset = ChessPGNDataset(config.pgn_paths, config)
    loader = DataLoader(
        dataset,
        batch_size=config.sft_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    best_loss = float('inf')
    
    for epoch in range(1, config.sft_epochs + 1):
        avg_loss, avg_p_loss, avg_v_loss = train_epoch(
            model, loader, optimizer, criterion, device, epoch, config.sft_epochs
        )
        
        print(f"Epoch {epoch}/{config.sft_epochs}: loss={avg_loss:.4f}")
        
        os.makedirs(config.base_path, exist_ok=True)
        save_checkpoint(
            model,
            optimizer,
            os.path.join(config.base_path, f'sft_epoch_{epoch}.pth'),
            metadata={
                'epoch': epoch,
                'loss': avg_loss,
                'policy_loss': avg_p_loss,
                'value_loss': avg_v_loss,
                'stage': 'sft'
            }
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model,
                optimizer,
                os.path.join(config.base_path, 'sft_best.pth'),
                metadata={
                    'epoch': epoch,
                    'loss': avg_loss,
                    'policy_loss': avg_p_loss,
                    'value_loss': avg_v_loss,
                    'stage': 'sft'
                }
            )
    
    from core.utils.evaluation import evaluate_vs_random
    model.eval()
    wins, losses, draws, winrate = evaluate_vs_random(model, config, num_games=100, device=device)
    print(f"Evaluation: {winrate:.1f}% win rate")


if __name__ == "__main__":
    main()
