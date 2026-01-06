"""
Self-Play Game Generation
"""
import chess
import numpy as np
import random
from typing import List, Tuple, Optional

from rl.mcts.search import MCTS
from core.chess_logic.board_encoding import BoardEncoder


def generate_self_play_game(
    mcts: MCTS,
    num_simulations: int = 100,
    temperature_threshold: int = 30,
    max_moves: int = 200
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, Optional[float], bool]], float, bool]:
    """
    Generate a single self-play game using MCTS
    
    Args:
        mcts: MCTS instance
        num_simulations: Number of MCTS simulations per move
        temperature_threshold: Move number after which temperature becomes 0
        max_moves: Maximum moves before declaring draw
    
    Returns:
        samples: List of (state, policy, value_placeholder, is_white_turn) tuples
        game_outcome: Final game outcome (1.0 = white win, -1.0 = black win, 0.0 = draw)
        white_perspective: Always True (for compatibility)
    """
    board = chess.Board()
    board_encoder = BoardEncoder()
    
    samples = []
    move_count = 0
    prev_board = None
    
    # Play game
    while not board.is_game_over() and move_count < max_moves:
        # Encode current state
        state = board_encoder.encode(board, prev_board)
        is_white_turn = (board.turn == chess.WHITE)
        
        # Determine temperature
        temperature = 1.0 if move_count < temperature_threshold else 0.0
        
        # Run MCTS
        policy_target = mcts.search(
            board, 
            num_simulations=num_simulations,
            temperature=temperature,
            prev_board=prev_board,
            root_noise=True
        )
        
        # Store sample (value will be filled in later)
        samples.append((state, policy_target, None, is_white_turn))
        
        # Select move
        legal_moves = list(board.legal_moves)
        move_encoder = mcts.move_encoder
        
        # Get probabilities for legal moves
        legal_probs = []
        for move in legal_moves:
            move_idx = move_encoder.encode_move(move.uci())
            if move_idx is not None:
                legal_probs.append(policy_target[move_idx])
            else:
                legal_probs.append(0.0)
        
        # Sample move
        if sum(legal_probs) > 0:
            legal_probs = np.array(legal_probs, dtype=np.float64)
            legal_probs = np.clip(legal_probs, 0.0, None)
            legal_probs /= legal_probs.sum()
            
            try:
                move_idx = np.random.choice(len(legal_moves), p=legal_probs)
                selected_move = legal_moves[move_idx]
            except:
                selected_move = random.choice(legal_moves)
        else:
            selected_move = random.choice(legal_moves)
        
        # Make move
        prev_board = board.copy()
        board.push(selected_move)
        move_count += 1
    
    # Determine game outcome
    if move_count >= max_moves:
        game_outcome = 0.0  # Draw by move limit
    else:
        result = board.result()
        if result == "1-0":
            game_outcome = 1.0
        elif result == "0-1":
            game_outcome = -1.0
        else:
            game_outcome = 0.0
    
    # Fill in values from game outcome
    final_samples = []
    for state, policy, _, was_white_turn in samples:
        # Value from perspective of player who made the move
        value = game_outcome if was_white_turn else -game_outcome
        final_samples.append((state, policy, value))
    
    return final_samples, game_outcome, True
