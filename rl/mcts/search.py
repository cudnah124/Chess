"""
MCTS Search - Monte Carlo Tree Search implementation
"""
import torch
import torch.nn as nn
import numpy as np
import chess
from typing import Optional

from .node import MCTSNode
from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance
    
    Args:
        model: Neural network model (SmallResNet)
        config: Configuration object
        device: PyTorch device
    """
    
    def __init__(self, model: nn.Module, config, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        self.move_encoder = CanonicalMoveEncoder()
        self.board_encoder = BoardEncoder()
        
        self.c_puct = config.c_puct
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_epsilon = config.dirichlet_epsilon
    
    def search(
        self, 
        board: chess.Board, 
        num_simulations: int,
        temperature: float = 1.0,
        prev_board: Optional[chess.Board] = None,
        root_noise: bool = True
    ) -> np.ndarray:
        """
        Run MCTS search from current position
        
        Args:
            board: Current board state
            num_simulations: Number of MCTS simulations
            temperature: Temperature for move selection (0 = greedy, 1 = stochastic)
            prev_board: Previous board state for history encoding
            root_noise: Whether to add Dirichlet noise to root
        
        Returns:
            action_probs: Probability distribution over all 4672 actions
        """
        root = MCTSNode()
        
        # Expand root node
        self._expand_node(root, board, prev_board)
        
        # Add Dirichlet noise to root for exploration
        if root_noise and len(root.children) > 0:
            self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(num_simulations):
            node = root
            search_board = board.copy()
            search_prev_board = prev_board
            
            # Selection: traverse tree to leaf
            while not node.is_leaf() and not search_board.is_game_over():
                node = node.select_child(c_puct=self.c_puct)
                if node.move:
                    search_prev_board = search_board.copy()
                    search_board.push(node.move)
            
            # Expansion & Evaluation
            if not search_board.is_game_over():
                value = self._expand_node(node, search_board, search_prev_board)
            else:
                # Terminal node
                value = self._get_terminal_value(search_board)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Extract action probabilities
        return self._get_action_probs(root, board, temperature)
    
    def _expand_node(self, node: MCTSNode, board: chess.Board, prev_board: Optional[chess.Board]) -> float:
        """
        Expand node using neural network
        
        Returns:
            value: Position evaluation from neural network
        """
        # Encode board state
        state = self.board_encoder.encode(board, prev_board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get neural network predictions
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        # Convert to probabilities
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()
        
        # Get legal moves and their priors
        legal_moves = list(board.legal_moves)
        moves_and_priors = {}
        total_prior = 0.0
        
        for move in legal_moves:
            move_idx = self.move_encoder.encode_move(move.uci())
            if move_idx is not None:
                prior = policy[move_idx]
            else:
                prior = 0.0
            moves_and_priors[move] = prior
            total_prior += prior
        
        # Normalize priors
        if total_prior > 0:
            moves_and_priors = {
                move: prior / total_prior 
                for move, prior in moves_and_priors.items()
            }
        else:
            # Fallback: uniform distribution
            uniform_prior = 1.0 / len(legal_moves)
            moves_and_priors = {move: uniform_prior for move in legal_moves}
        
        # Expand node
        node.expand(moves_and_priors)
        
        return value
    
    def _get_terminal_value(self, board: chess.Board) -> float:
        """
        Get value for terminal position
        
        Returns:
            value: 1.0 for win, -1.0 for loss, 0.0 for draw (from perspective of side to move)
        """
        result = board.result()
        
        if result == "1-0":  # White wins
            return -1.0 if board.turn == chess.BLACK else 1.0
        elif result == "0-1":  # Black wins
            return -1.0 if board.turn == chess.WHITE else 1.0
        else:  # Draw
            return 0.0
    
    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root node for exploration"""
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
        
        for (move, child), n in zip(root.children.items(), noise):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * n
    
    def _get_action_probs(self, root: MCTSNode, board: chess.Board, temperature: float) -> np.ndarray:
        """
        Extract action probabilities from visit counts
        
        Args:
            root: Root node
            board: Current board
            temperature: Temperature parameter (0 = greedy, 1 = stochastic)
        
        Returns:
            action_probs: Probability distribution over 4672 actions
        """
        action_probs = np.zeros(4672, dtype=np.float32)
        
        # Get visit counts
        visit_counts = root.get_visit_counts()
        
        if len(visit_counts) == 0:
            return action_probs
        
        # Convert to numpy arrays
        moves = list(visit_counts.keys())
        visits = np.array([visit_counts[m] for m in moves], dtype=np.float32)
        
        # Apply temperature
        if temperature == 0:
            # Greedy: select most visited
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            # Stochastic: proportional to visits^(1/T)
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
        
        # Map to action space
        for move, prob in zip(moves, probs):
            move_idx = self.move_encoder.encode_move(move.uci())
            if move_idx is not None:
                action_probs[move_idx] = prob
        
        return action_probs
