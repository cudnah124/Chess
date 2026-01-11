"""
Test: Evaluate trained model vs Minimax AI
"""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import torch
import chess
import random
import numpy as np
from core.config import Config
from core.models.resnet import SmallResNet
from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder
from core.utils.checkpoint import load_checkpoint
from rl.mcts.search import MCTS
from rl.mcts.search import MCTS


class MinimaxPlayer:
    """Simple Minimax chess player"""
    
    def __init__(self, depth=3):
        self.depth = depth
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def evaluate_board(self, board):
        """Evaluate board position"""
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        return score
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        """Minimax with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_board(board)
        
        # Move ordering: prioritize captures for better pruning
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        
        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_move(self, board):
        """Get best move using Minimax"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Add randomization and move ordering (captures first)
        random.shuffle(legal_moves)
        legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        
        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        for move in legal_moves:
            board.push(move)
            value = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), 
                               board.turn == chess.WHITE)
            board.pop()
            
            if board.turn == chess.WHITE:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
       


class NeuralPlayer:
    """Neural network chess player with MCTS"""
    
    def __init__(self, model, config, device='cpu', use_mcts=True, num_simulations=50, temperature=0.1):
        """
        Args:
            model: Neural network model
            config: Configuration object
            device: Device to run on
            use_mcts: Whether to use MCTS (True) or pure policy (False)
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection (lower = more deterministic)
        """
        self.model = model
        self.config = config
        self.device = device
        self.move_encoder = CanonicalMoveEncoder()
        self.board_encoder = BoardEncoder()
        self.model.eval()
        
        # MCTS configuration
        self.use_mcts = use_mcts
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        if use_mcts:
            self.mcts = MCTS(model, config, device)
    
    def get_move(self, board, prev_board=None):
        """Get move from neural network (with or without MCTS)"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if self.use_mcts:
            # Use MCTS for stronger play
            action_probs = self.mcts.search(
                board=board,
                num_simulations=self.num_simulations,
                temperature=self.temperature,
                prev_board=prev_board,
                root_noise=False  # No exploration noise during evaluation
            )
            
            # Select move with highest probability
            move_probs = []
            for move in legal_moves:
                move_idx = self.move_encoder.encode_move(move.uci())
                prob = action_probs[move_idx] if move_idx is not None else 0.0
                move_probs.append((move, prob))
            
            if sum(p for _, p in move_probs) > 0:
                move_probs.sort(key=lambda x: x[1], reverse=True)
                return move_probs[0][0]
            else:
                return random.choice(legal_moves)
        else:
            # Pure policy network (weaker)
            state = self.board_encoder.encode(board, prev_board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, _ = self.model(state_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            move_probs = []
            for move in legal_moves:
                move_idx = self.move_encoder.encode_move(move.uci())
                prob = policy[move_idx] if move_idx is not None else 0.0
                move_probs.append((move, prob))
            
            if sum(p for _, p in move_probs) > 0:
                move_probs.sort(key=lambda x: x[1], reverse=True)
                return move_probs[0][0]
            else:
                return random.choice(legal_moves)


def play_game(white_player, black_player, max_moves=200, verbose=False):
    """Play a game between two players"""
    board = chess.Board()
    prev_board = None
    move_count = 0
    
    if verbose:
        print(f"\n{board}")
    
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            if isinstance(white_player, NeuralPlayer):
                move = white_player.get_move(board, prev_board)
            else:
                move = white_player.get_move(board)
        else:
            if isinstance(black_player, NeuralPlayer):
                move = black_player.get_move(board, prev_board)
            else:
                move = black_player.get_move(board)
        
        if move is None:
            break
        
        if verbose:
            player_name = "White (Neural)" if board.turn == chess.WHITE and isinstance(white_player, NeuralPlayer) else \
                         "White (Minimax)" if board.turn == chess.WHITE else \
                         "Black (Neural)" if isinstance(black_player, NeuralPlayer) else "Black (Minimax)"
            print(f"\nMove {move_count + 1}: {player_name} plays {move}")
        
        prev_board = board.copy()
        board.push(move)
        move_count += 1
        
        if verbose:
            print(f"{board}")
    
    if verbose:
        print(f"\nGame over after {move_count} moves")
        print(f"Result: {board.result()}")
    
    if move_count >= max_moves:
        return 0.5  # Draw
    
    result = board.result()
    if result == "1-0":
        return 1.0  # White wins
    elif result == "0-1":
        return 0.0  # Black wins
    else:
        return 0.5  # Draw


def evaluate_vs_minimax(checkpoint_path, num_games=20, minimax_depth=3, 
                        use_mcts=True, num_simulations=50, temperature=0.1, verbose=False):
    """
    Evaluate trained model vs Minimax
    """
    print(f"Loading model from: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    model = SmallResNet(config).to(device)
    load_checkpoint(model, checkpoint_path, expected_action_size=4672, device=device)
    
    neural_player = NeuralPlayer(
        model, config, device, 
        use_mcts=use_mcts, 
        num_simulations=num_simulations, 
        temperature=temperature
    )
    minimax_player = MinimaxPlayer(depth=minimax_depth)
    
    mcts_status = f"with MCTS ({num_simulations} sims)" if use_mcts else "pure policy (no MCTS)"
    print(f"\nEvaluating: {num_games} games vs Minimax (depth={minimax_depth})")
    print(f"Neural player mode: {mcts_status}")
    print(f"{'='*60}")
    
    # Neural as White
    print(f"\nNeural (White) vs Minimax (Black):")
    white_wins = 0
    for i in range(num_games // 2):
        if verbose:
            print(f"\n{'='*60}")
            print(f"GAME {i+1}: Neural (White) vs Minimax (Black)")
            print(f"{'='*60}")
        result = play_game(neural_player, minimax_player, verbose=verbose)
        if result == 1.0:
            white_wins += 1
        if not verbose:
            print(f"  Game {i+1}: {'Win' if result == 1.0 else ('Loss' if result == 0.0 else 'Draw')}")
    
    # Neural as Black
    print(f"\nMinimax (White) vs Neural (Black):")
    black_wins = 0
    for i in range(num_games // 2):
        if verbose:
            print(f"\n{'='*60}")
            print(f"GAME {i+1}: Minimax (White) vs Neural (Black)")
            print(f"{'='*60}")
        result = play_game(minimax_player, neural_player, verbose=verbose)
        if result == 0.0:
            black_wins += 1
        if not verbose:
            print(f"  Game {i+1}: {'Win' if result == 0.0 else ('Loss' if result == 1.0 else 'Draw')}")
    
    total_wins = white_wins + black_wins
    winrate = (total_wins / num_games) * 100
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Neural wins: {total_wins}/{num_games}")
    print(f"  Win rate: {winrate:.1f}%")
    print(f"{'='*60}")
    
    return winrate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model vs Minimax')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=20, help='Number of games')
    parser.add_argument('--depth', type=int, default=2, help='Minimax search depth')
    parser.add_argument('--no-mcts', action='store_true', help='Disable MCTS (use pure policy)')
    parser.add_argument('--simulations', type=int, default=50, help='Number of MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for move selection')
    parser.add_argument('--verbose', action='store_true', help='Show moves during games')
    
    args = parser.parse_args()
    
    evaluate_vs_minimax(
        args.checkpoint, 
        args.games, 
        args.depth, 
        use_mcts=not args.no_mcts,
        num_simulations=args.simulations,
        temperature=args.temperature,
        verbose=args.verbose
    )

