"""
Board State Encoding - Convert chess.Board to 32-channel tensor
"""
import chess
import numpy as np
from typing import Optional


class BoardEncoder:
    """
    Encodes chess board state as 32-channel 8×8 tensor:
    
    Channels 0-11: Current piece positions (6 types × 2 colors)
    Channels 12-23: Previous position (T-1 history)
    Channels 24-31: Auxiliary features (turn, castling, en passant, repetition)
    """
    
    def __init__(self):
        self.piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
    
    def encode(self, board: chess.Board, prev_board: Optional[chess.Board] = None) -> np.ndarray:
        """
        Encode board state to 32×8×8 tensor
        
        Args:
            board: Current board state
            prev_board: Previous board state (for history channel)
        
        Returns:
            np.ndarray of shape (32, 8, 8) with dtype float32
        """
        tensor = np.zeros((32, 8, 8), dtype=np.float32)
        
        # Channels 0-11: Current piece positions
        for square, piece in board.piece_map().items():
            channel = self.piece_map[piece.symbol()]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[channel][rank][file] = 1.0
        
        # Channels 12-23: Previous position (history T-1)
        if prev_board is not None:
            for square, piece in prev_board.piece_map().items():
                channel = self.piece_map[piece.symbol()] + 12
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                tensor[channel][rank][file] = 1.0
        
        # Channel 24: Turn (1 if white to move, 0 if black)
        if board.turn == chess.WHITE:
            tensor[24, :, :] = 1.0
        
        # Channel 25: White kingside castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[25, :, :] = 1.0
        
        # Channel 26: White queenside castling rights
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[26, :, :] = 1.0
        
        # Channel 27: Black kingside castling rights
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[27, :, :] = 1.0
        
        # Channel 28: Black queenside castling rights
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[28, :, :] = 1.0
        
        # Channel 29: En passant square
        if board.ep_square is not None:
            rank = chess.square_rank(board.ep_square)
            file = chess.square_file(board.ep_square)
            tensor[29][rank][file] = 1.0
        
        # Channel 30: Repetition (1-fold)
        if board.is_repetition(1):
            tensor[30, :, :] = 1.0
        
        # Channel 31: Repetition (2-fold)
        if board.is_repetition(2):
            tensor[31, :, :] = 1.0
        
        return tensor
