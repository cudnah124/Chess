"""
Evaluation CLI - Test model performance
"""
import sys
import os
import torch
import argparse

from core.config import Config
from core.models.resnet import SmallResNet
from core.utils.checkpoint import load_checkpoint
from core.utils.evaluation import evaluate_vs_random


def main():
    parser = argparse.ArgumentParser(description='Evaluate Chess AI model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--opponent', type=str, default='random', choices=['random'], help='Opponent type')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--config', type=str, default='configs/rl_config.yaml', help='Config file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f" Device: {device}")
    
    # Load config
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        print(f" Config not found, using defaults")
        config = Config()
    
    # Load model
    print(f"\n🔄 Loading model: {args.checkpoint}")
    model = SmallResNet(config).to(device)
    
    try:
        metadata = load_checkpoint(
            model,
            args.checkpoint,
            expected_action_size=config.ACTION_SIZE,
            device=device
        )
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"🎮 Evaluating vs {args.opponent.upper()}")
    print(f"{'='*60}")
    print(f"Games: {args.games}")
    
    model.eval()
    
    if args.opponent == 'random':
        wins, losses, draws, winrate = evaluate_vs_random(
            model,
            config,
            num_games=args.games,
            device=device
        )
    
    # Results
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Wins:     {wins:4d} ({wins/args.games*100:5.1f}%)")
    print(f"Losses:   {losses:4d} ({losses/args.games*100:5.1f}%)")
    print(f"Draws:    {draws:4d} ({draws/args.games*100:5.1f}%)")
    print(f"{'='*60}")
    print(f"Win Rate: {winrate:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
