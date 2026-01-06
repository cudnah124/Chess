"""
Replay Buffer - Experience replay for RL training
"""
import pickle
import os
import random
from collections import deque
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    """
    Circular buffer for storing self-play experiences
    
    Each sample is a tuple: (state, policy_target, value_target)
    - state: np.ndarray of shape (32, 8, 8)
    - policy_target: np.ndarray of shape (4672,)
    - value_target: float
    """
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_batch(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Add a batch of samples to buffer
        
        Args:
            samples: List of (state, policy, value) tuples
        """
        self.buffer.extend(samples)
    
    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Sample a random batch from buffer
        
        Args:
            batch_size: Number of samples to return
        
        Returns:
            List of (state, policy, value) tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str):
        """Save buffer to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path: str) -> bool:
        """
        Load buffer from disk
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data, maxlen=self.max_size)
            return True
        except Exception as e:
            return False
    
    def clear(self):
        """Clear all samples from buffer"""
        self.buffer.clear()
