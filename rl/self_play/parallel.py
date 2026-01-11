import torch
import numpy as np
import random
import chess
from multiprocessing import Pool
from tqdm import tqdm

from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder
from core.models.resnet import SmallResNet


def worker_play_game(args):
    model_path, seed, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, num_res_blocks, num_channels, action_size = args
    
    torch.set_num_threads(1)
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    model = SmallResNet(num_res_blocks=num_res_blocks, num_channels=num_channels, action_size=action_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    from rl.mcts.search import SimpleMCTS
    
    mcts = SimpleMCTS(model, c_puct, dirichlet_alpha, dirichlet_epsilon)
    move_encoder = CanonicalMoveEncoder()
    board_encoder = BoardEncoder()
    
    board = chess.Board()
    samples = []
    prev_board = None
    move_count = 0
    max_moves = 80
    
    while not board.is_game_over() and move_count < max_moves:
        state = board_encoder.encode(board, prev_board)
        is_white = (board.turn == chess.WHITE)
        policy = mcts.search(board, num_simulations, prev_board, add_noise=True)
        samples.append((state, policy, None, is_white))
        
        legal_moves = list(board.legal_moves)
        legal_probs = []
        for move in legal_moves:
            move_idx = move_encoder.encode_move(move.uci())
            legal_probs.append(policy[move_idx] if move_idx is not None else 0.0)
        
        if sum(legal_probs) > 0:
            legal_probs = np.array(legal_probs, dtype=np.float64)
            legal_probs /= legal_probs.sum()
            move_idx = np.random.choice(len(legal_moves), p=legal_probs)
            selected_move = legal_moves[move_idx]
        else:
            selected_move = random.choice(legal_moves)
        
        prev_board = board.copy()
        board.push(selected_move)
        move_count += 1
    
    if move_count >= max_moves:
        outcome = 0.0
    else:
        result = board.result()
        outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    
    final_samples = []
    for state, policy, _, was_white in samples:
        value = outcome if was_white else -outcome
        final_samples.append((state, policy, value))
    
    return final_samples


def parallel_self_play(model, num_games, num_simulations, num_workers, c_puct, dirichlet_alpha, dirichlet_epsilon, temp_model_path, num_res_blocks, num_channels, action_size):
    torch.save(model.state_dict(), temp_model_path)
    
    worker_args = [
        (temp_model_path, seed, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, num_res_blocks, num_channels, action_size)
        for seed in range(num_games)
    ]
    
    all_samples = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_play_game, worker_args), 
            total=num_games, 
            desc="Self-play games"
        ))
    
    for samples in results:
        all_samples.extend(samples)
    
    return all_samples
