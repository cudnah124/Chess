"""
Chess PGN Dataset - PyTorch Dataset for supervised training
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple

from sft.data.preprocessing import process_multiple_pgn_files


class ChessPGNDataset(Dataset):
    """
    PyTorch Dataset for chess positions from PGN files
    
    Args:
        pgn_paths: List of PGN file paths
        config: Configuration object
        preload: Whether to preload all data into memory
    """
    
    def __init__(self, pgn_paths: List[str], config, preload: bool = True):
        self.config = config
        
        # Process PGN files
        print(f"\n{'='*60}")
        print(f"📦 Loading PGN Dataset")
        print(f"{'='*60}")
        
        max_games_per_file = config.max_games // len(pgn_paths) if config.max_games else None
        
        self.samples = process_multiple_pgn_files(
            pgn_paths,
            max_games_per_file=max_games_per_file
        )
        
        print(f"\n Dataset ready: {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample
        
        Returns:
            dict with keys: 'state', 'policy', 'value'
        """
        state, policy, value = self.samples[idx]
        
        return {
            'state': torch.FloatTensor(state),      # (32, 8, 8)
            'policy': torch.FloatTensor(policy),    # (4672,)
            'value': torch.FloatTensor([value])     # (1,)
        }
