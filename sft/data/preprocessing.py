"""
PGN Data Processing - Extract training samples from chess games
"""
import chess
import chess.pgn
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder


def process_pgn_file(
    pgn_path: str,
    max_games: Optional[int] = None,
    verbose: bool = True
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Process PGN file and extract training samples
    
    Args:
        pgn_path: Path to PGN file
        max_games: Maximum number of games to process
        verbose: Whether to show progress bar
    
    Returns:
        List of (state, policy_target, value_target) tuples
    """
    move_encoder = CanonicalMoveEncoder()
    board_encoder = BoardEncoder()
    
    samples = []
    games_processed = 0
    games_skipped = 0
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(desc=f"Processing {pgn_path}", disable=not verbose)
        
        while True:
            if max_games and games_processed >= max_games:
                break
            
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            # Get game result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_outcome = 1.0
            elif result == "0-1":
                game_outcome = -1.0
            elif result == "1/2-1/2":
                game_outcome = 0.0
            else:
                games_skipped += 1
                continue
            
            # Extract samples from game
            board = game.board()
            prev_board = None
            
            for move in game.mainline_moves():
                # Encode current state
                state = board_encoder.encode(board, prev_board)
                
                # Encode move as one-hot policy target
                move_idx = move_encoder.encode_move(move.uci())
                
                if move_idx is None:
                    # Move not in canonical space, skip this position
                    prev_board = board.copy()
                    board.push(move)
                    continue
                
                # Create one-hot policy target
                policy_target = np.zeros(4672, dtype=np.float32)
                policy_target[move_idx] = 1.0
                
                # Value target from perspective of current player
                value_target = game_outcome if board.turn == chess.WHITE else -game_outcome
                
                # Add sample
                samples.append((state, policy_target, value_target))
                
                # Update board
                prev_board = board.copy()
                board.push(move)
            
            games_processed += 1
            pbar.update(1)
            pbar.set_postfix({
                'games': games_processed,
                'samples': len(samples),
                'skipped': games_skipped
            })
        
        pbar.close()
    
    if verbose:
        print(f" Processed {games_processed} games")
        print(f"   Total samples: {len(samples)}")
        print(f"   Skipped games: {games_skipped}")
    
    return samples


def process_multiple_pgn_files(
    pgn_paths: List[str],
    max_games_per_file: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Process multiple PGN files
    
    Args:
        pgn_paths: List of PGN file paths
        max_games_per_file: Maximum games per file
    
    Returns:
        Combined list of samples
    """
    all_samples = []
    
    for pgn_path in pgn_paths:
        print(f"\n📂 Processing: {pgn_path}")
        samples = process_pgn_file(pgn_path, max_games=max_games_per_file)
        all_samples.extend(samples)
    
    print(f"\n Total samples from all files: {len(all_samples)}")
    return all_samples
