import torch
import torch.nn as nn
import numpy as np
import chess
import random

from .node import MCTSNode
from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder


class SimpleMCTS:
    def __init__(self, model, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.move_encoder = CanonicalMoveEncoder()
        self.board_encoder = BoardEncoder()
    
    def search(self, board, num_simulations, prev_board=None, add_noise=True):
        root = MCTSNode(0)
        
        state = torch.FloatTensor(self.board_encoder.encode(board, prev_board)).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
        
        for move in board.legal_moves:
            move_idx = self.move_encoder.encode_move(move.uci())
            if move_idx is not None:
                root.children[move.uci()] = MCTSNode(policy[move_idx])
        
        for _ in range(num_simulations):
            self._simulate(root, board.copy(), prev_board)
        
        policy_out = np.zeros(4672, dtype=np.float32)
        for move_uci, child in root.children.items():
            move_idx = self.move_encoder.encode_move(move_uci)
            if move_idx is not None:
                policy_out[move_idx] = child.visit_count
        
        if policy_out.sum() > 0:
            policy_out /= policy_out.sum()
        
        return policy_out
    
    def _simulate(self, node, board, prev_board):
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0 if board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                return -1.0 if board.turn == chess.BLACK else 1.0
            return 0.0
        
        if not node.children:
            state = torch.FloatTensor(self.board_encoder.encode(board, prev_board)).unsqueeze(0)
            with torch.no_grad():
                _, value = self.model(state)
            return value.item() * (1 if board.turn == chess.WHITE else -1)
        
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        total_visits = sum(child.visit_count for child in node.children.values())
        
        for move_uci, child in node.children.items():
            q_value = -child.value()
            u_value = self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_move = chess.Move.from_uci(move_uci)
                best_child = child
        
        prev_for_next = board.copy()
        board.push(best_move)
        value = self._simulate(best_child, board, prev_for_next)
        
        best_child.visit_count += 1
        best_child.value_sum += -value
        
        return -value


class MCTS:
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        self.move_encoder = CanonicalMoveEncoder()
        self.board_encoder = BoardEncoder()
        
        self.c_puct = getattr(config, 'c_puct', getattr(config, 'rl_c_puct', 1.5))
        self.dirichlet_alpha = getattr(config, 'dirichlet_alpha', getattr(config, 'rl_dirichlet_alpha', 0.3))
        self.dirichlet_epsilon = getattr(config, 'dirichlet_epsilon', getattr(config, 'rl_dirichlet_epsilon', 0.25))
    
    def search(self, board, num_simulations, temperature=1.0, prev_board=None, root_noise=True):
        root = MCTSNode(0)
        
        state = self.board_encoder.encode(board, prev_board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
        
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
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
        
        if total_prior > 0:
            moves_and_priors = {move: prior / total_prior for move, prior in moves_and_priors.items()}
        else:
            uniform_prior = 1.0 / len(legal_moves)
            moves_and_priors = {move: uniform_prior for move in legal_moves}
        
        for move, prior in moves_and_priors.items():
            root.children[move.uci()] = MCTSNode(prior)
        
        if root_noise and len(root.children) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for (move_uci, child), n in zip(root.children.items(), noise):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * n
        
        for _ in range(num_simulations):
            self._simulate(root, board.copy(), prev_board)
        
        action_probs = np.zeros(4672, dtype=np.float32)
        
        visit_counts = {move_uci: child.visit_count for move_uci, child in root.children.items()}
        
        if len(visit_counts) == 0:
            return action_probs
        
        moves = list(visit_counts.keys())
        visits = np.array([visit_counts[m] for m in moves], dtype=np.float32)
        
        if temperature == 0:
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
        
        for move_uci, prob in zip(moves, probs):
            move_idx = self.move_encoder.encode_move(move_uci)
            if move_idx is not None:
                action_probs[move_idx] = prob
        
        return action_probs
    
    def _simulate(self, node, board, prev_board):
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return -1.0 if board.turn == chess.BLACK else 1.0
            elif result == "0-1":
                return -1.0 if board.turn == chess.WHITE else 1.0
            return 0.0
        
        if not node.children:
            state = self.board_encoder.encode(board, prev_board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, value = self.model(state_tensor)
            
            return value.cpu().item()
        
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        total_visits = sum(child.visit_count for child in node.children.values())
        
        for move_uci, child in node.children.items():
            q_value = -child.value()
            u_value = self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_move = chess.Move.from_uci(move_uci)
                best_child = child
        
        prev_for_next = board.copy()
        board.push(best_move)
        value = self._simulate(best_child, board, prev_for_next)
        
        best_child.visit_count += 1
        best_child.value_sum += -value
        
        return -value
