"""
MCTS Node - Tree node for Monte Carlo Tree Search
"""
import numpy as np
from typing import Optional, Dict
import chess


class MCTSNode:
    """
    Node in the MCTS tree
    
    Attributes:
        parent: Parent node
        move: Move that led to this node
        prior: Prior probability from neural network
        children: Dictionary of child nodes (move -> node)
        visit_count: Number of times this node was visited
        value_sum: Sum of values from all visits
    """
    
    def __init__(self, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None, prior: float = 0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
    
    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node"""
        return self.parent is None
    
    def select_child(self, c_puct: float = 1.5, fpu_mode: str = 'parent') -> 'MCTSNode':
        """
        Select best child using UCB formula
        
        Args:
            c_puct: Exploration constant
            fpu_mode: First Play Urgency mode ('parent' or 'zero')
        
        Returns:
            Best child node
        """
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            # Q-value (exploitation)
            if child.visit_count == 0:
                # First Play Urgency
                q_value = self.value() if fpu_mode == 'parent' else 0.0
            else:
                q_value = -child.value()  # Negamax
            
            # U-value (exploration)
            u_value = c_puct * child.prior * np.sqrt(max(1, self.visit_count)) / (1 + child.visit_count)
            
            # UCB score
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else list(self.children.values())[0]
    
    def expand(self, moves_and_priors: Dict[chess.Move, float]):
        """
        Expand node with children
        
        Args:
            moves_and_priors: Dictionary of legal moves and their prior probabilities
        """
        for move, prior in moves_and_priors.items():
            if move not in self.children:
                self.children[move] = MCTSNode(parent=self, move=move, prior=prior)
    
    def backpropagate(self, value: float):
        """
        Backpropagate value up the tree
        
        Args:
            value: Value to backpropagate (from perspective of current player)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent
    
    def get_visit_counts(self) -> Dict[chess.Move, int]:
        """Get visit counts for all children"""
        return {move: child.visit_count for move, child in self.children.items()}
    
    def __repr__(self):
        return f"MCTSNode(move={self.move}, visits={self.visit_count}, value={self.value():.3f}, prior={self.prior:.3f})"
