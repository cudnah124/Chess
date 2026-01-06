"""
Evaluation Utilities - Test model performance
"""
import torch
import torch.nn as nn
import chess
import random
import numpy as np
from tqdm import tqdm
from typing import Tuple

from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder


def evaluate_vs_random(
    model: nn.Module,
    config,
    num_games: int = 100,
    device: str = 'cpu'
) -> Tuple[int, int, int, float]:
    """
    Evaluate model against random player
    
    Args:
        model: Neural network model
        config: Configuration object
        num_games: Number of games to play
        device: PyTorch device
    
    Returns:
        wins, losses, draws, winrate
    """
    model.eval()
    move_encoder = CanonicalMoveEncoder()
    board_encoder = BoardEncoder()
    
    wins, losses, draws = 0, 0, 0
    
    for _ in tqdm(range(num_games), desc="Evaluating"):
        board = chess.Board()
        ai_plays_white = random.choice([True, False])
        prev_board = None
        move_count = 0
        
        while not board.is_game_over() and move_count < 200:
            if (board.turn == chess.WHITE) == ai_plays_white:
                # AI move
                state = board_encoder.encode(board, prev_board)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    policy_logits, _ = model(state_tensor)
                
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                
                # Get legal moves
                legal_moves = list(board.legal_moves)
                legal_probs = []
                for move in legal_moves:
                    move_idx = move_encoder.encode_move(move.uci())
                    if move_idx is not None:
                        legal_probs.append(policy[move_idx])
                    else:
                        legal_probs.append(0.0)
                
                # Select best move
                if sum(legal_probs) > 0:
                    chosen = legal_moves[np.argmax(legal_probs)]
                else:
                    chosen = random.choice(legal_moves)
                
                prev_board = board.copy()
                board.push(chosen)
            else:
                # Random move
                prev_board = board.copy()
                board.push(random.choice(list(board.legal_moves)))
            
            move_count += 1
        
        # Determine outcome
        result = board.result()
        if result == "1/2-1/2":
            draws += 1
        elif result == "1-0":
            if ai_plays_white:
                wins += 1
            else:
                losses += 1
        else:  # "0-1"
            if ai_plays_white:
                losses += 1
            else:
                wins += 1
    
    winrate = (wins / num_games) * 100 if num_games > 0 else 0.0
    return wins, losses, draws, winrate
