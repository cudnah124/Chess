"""
RL Training Script - Main entry point for reinforcement learning
"""
import torch
import os
import sys

from core.config import Config
from core.models.resnet import SmallResNet
from core.utils.checkpoint import load_checkpoint, save_checkpoint
from rl.trainer import AlphaZeroTrainer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/rl_config.yaml'
    config = Config.from_yaml(config_path)
    config.validate()
    
    model = SmallResNet(config).to(device)
    
    if config.sft_checkpoint_path and os.path.exists(config.sft_checkpoint_path):
        load_checkpoint(
            model,
            config.sft_checkpoint_path,
            expected_action_size=config.ACTION_SIZE,
            device=device
        )
    
    trainer = AlphaZeroTrainer(model, config, device)
    
    buffer_path = os.path.join(config.base_path, 'buffer_rl_latest.pkl')
    if os.path.exists(buffer_path):
        trainer.replay_buffer.load(buffer_path)
    
    best_winrate = 0.0
    
    for iteration in range(config.rl_iterations):
        print(f"Iteration {iteration + 1}/{config.rl_iterations}")
        
        eval_this_iter = (iteration + 1) % config.eval_interval == 0
        wins, losses, draws, winrate, buffer_size = trainer.train_iteration(
            num_games=config.games_per_iter,
            num_simulations=config.rl_num_simulations,
            eval_vs_random=eval_this_iter
        )
        
        save_checkpoint(
            model,
            trainer.optimizer,
            os.path.join(config.base_path, f'rl_iter_{iteration+1}.pth'),
            metadata={
                'iteration': iteration + 1,
                'winrate': winrate if eval_this_iter else None,
                'buffer_size': buffer_size,
                'stage': 'rl'
            }
        )
        
        if eval_this_iter and winrate > best_winrate:
            best_winrate = winrate
            save_checkpoint(
                model,
                trainer.optimizer,
                os.path.join(config.base_path, 'rl_best.pth'),
                metadata={
                    'iteration': iteration + 1,
                    'winrate': winrate,
                    'stage': 'rl'
                }
            )
        
        trainer.replay_buffer.save(buffer_path)


if __name__ == "__main__":
    main()
