#ChessConverter - Board encoding & Move mapping (Hard-coded)
from config import *
class ChessConverter:
    def __init__(self, move_to_idx=None, idx_to_move=None):
        self.piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        # Use fixed canonical map; no dynamic additions
        self.move_to_idx = move_to_idx if move_to_idx is not None else CANONICAL_MOVE_TO_IDX
        self.idx_to_move = idx_to_move if idx_to_move is not None else CANONICAL_IDX_TO_MOVE
        self.next_idx = len(self.move_to_idx)  # informational only

    def encode_move(self, move_uci):
        # Hard-coded: unknown moves are ignored
        idx = self.move_to_idx.get(move_uci, None)
        if idx is not None:
            return idx
        # This preserves the 4672 action space and supports common promotions.
        if isinstance(move_uci, str) and len(move_uci) == 5 and move_uci[-1] == 'q':
            base_uci = move_uci[:4]
            return self.move_to_idx.get(base_uci, None)
        return None

    def board_to_tensor(self, board, prev_board=None):
        tensor = np.zeros((32, 8, 8), dtype=np.float32)
        for sq, pc in board.piece_map().items():
            tensor[self.piece_map[pc.symbol()]][chess.square_rank(sq)][chess.square_file(sq)] = 1
        if prev_board:
            for sq, pc in prev_board.piece_map().items():
                tensor[self.piece_map[pc.symbol()]+12][chess.square_rank(sq)][chess.square_file(sq)] = 1
        if board.turn == chess.WHITE: tensor[24,:,:] = 1
        if board.has_kingside_castling_rights(chess.WHITE): tensor[25,:,:] = 1
        if board.has_queenside_castling_rights(chess.WHITE): tensor[26,:,:] = 1
        if board.has_kingside_castling_rights(chess.BLACK): tensor[27,:,:] = 1
        if board.has_queenside_castling_rights(chess.BLACK): tensor[28,:,:] = 1
        if board.ep_square:
            tensor[29][chess.square_rank(board.ep_square)][chess.square_file(board.ep_square)] = 1
        if board.is_repetition(1): tensor[30,:,:] = 1
        if board.is_repetition(2): tensor[31,:,:] = 1
        return tensor