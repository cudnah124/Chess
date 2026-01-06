"""
Canonical Move Encoding - Fixed 4672 action space for Chess
"""
import chess
from typing import Dict, Optional, Tuple


class CanonicalMoveEncoder:
    """
    Fixed 4672 action space following AlphaZero design:
    - Queen moves (8 directions × 7 distances × 64 squares): 3584 indices
    - Knight moves (8 directions × 64 squares): 512 indices  
    - Underpromotions (3 directions × 3 pieces × 64 squares): 576 indices
    
    Total: 3584 + 512 + 576 = 4672
    
    Note: Queen promotions are handled as regular moves without promotion suffix
    """
    
    def __init__(self):
        self.move_to_idx, self.idx_to_move = self._build_canonical_map()
    
    def _build_canonical_map(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build fixed 4672 action space"""
        move_to_idx = {}
        idx_to_move = {}
        idx = 0
        
        # 1. Queen Moves (8 directions × 7 distances × 64 squares = 3584 slots)
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for from_sq in range(64):
            from_rank = from_sq // 8
            from_file = from_sq % 8
            for d_r, d_f in directions:
                for dist in range(1, 8):
                    to_rank = from_rank + d_r * dist
                    to_file = from_file + d_f * dist
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        dest = to_rank * 8 + to_file
                        uci = chess.Move(from_sq, dest).uci()
                        move_to_idx[uci] = idx
                        idx_to_move[idx] = uci
                    idx += 1  # Always increment to maintain 4672 structure
        
        # 2. Knight Moves (8 directions × 64 squares = 512 slots)
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        for from_sq in range(64):
            from_rank = from_sq // 8
            from_file = from_sq % 8
            for d_r, d_f in knight_moves:
                to_rank = from_rank + d_r
                to_file = from_file + d_f
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    dest = to_rank * 8 + to_file
                    uci = chess.Move(from_sq, dest).uci()
                    move_to_idx[uci] = idx
                    idx_to_move[idx] = uci
                idx += 1  # Always increment to maintain 4672 structure
        
        # 3. Underpromotions (3 file steps × 3 pieces × 64 squares = 576 slots)
        file_steps = [0, -1, 1]
        promotions = ['r', 'b', 'n']
        
        for from_sq in range(64):
            rank = from_sq // 8
            file = from_sq % 8
            
            # Determine promotion direction
            rank_step = 0
            if rank == 6:  # White pawn on 7th rank
                rank_step = 1
            elif rank == 1:  # Black pawn on 2nd rank
                rank_step = -1
            
            for f_step in file_steps:
                for p in promotions:
                    if rank_step != 0:
                        to_rank = rank + rank_step
                        to_file = file + f_step
                        if 0 <= to_file < 8:
                            dest = to_rank * 8 + to_file
                            uci = chess.Move(
                                from_sq, dest, 
                                promotion=chess.Piece.from_symbol(p).piece_type
                            ).uci()
                            move_to_idx[uci] = idx
                            idx_to_move[idx] = uci
                    idx += 1  # Always increment to maintain 4672 structure
        
        assert idx == 4672, f"Expected 4672 slots, got {idx}"
        return move_to_idx, idx_to_move
    
    def encode_move(self, move_uci: str) -> Optional[int]:
        """
        Encode UCI move string to index
        
        Args:
            move_uci: UCI format move (e.g., 'e2e4', 'e7e8q')
        
        Returns:
            Index in [0, 4672) or None if move not in canonical space
        """
        # Direct lookup
        idx = self.move_to_idx.get(move_uci, None)
        if idx is not None:
            return idx
        
        # Fallback: Queen promotions (e.g., 'e7e8q' -> 'e7e8')
        if isinstance(move_uci, str) and len(move_uci) == 5 and move_uci[-1] == 'q':
            base_uci = move_uci[:4]
            return self.move_to_idx.get(base_uci, None)
        
        return None
    
    def decode_move(self, idx: int) -> Optional[str]:
        """
        Decode index to UCI move string
        
        Args:
            idx: Index in [0, 4672)
        
        Returns:
            UCI format move string or None if invalid index
        """
        return self.idx_to_move.get(idx, None)
    
    @property
    def action_size(self) -> int:
        """Always returns 4672 (fixed action space)"""
        return 4672
    
    def is_valid_move(self, move_uci: str) -> bool:
        """Check if move is in canonical action space"""
        return self.encode_move(move_uci) is not None
