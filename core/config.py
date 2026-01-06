"""
Core configuration module - Single source of truth for all hyperparameters
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class Config:
    """
    Unified configuration for Chess AI training pipeline
    
    Architecture parameters are IMMUTABLE to ensure dimension consistency
    between SFT and RL stages.
    """
    
    # ==================== Model Architecture (IMMUTABLE) ====================
    NUM_RES_BLOCKS: int = 6
    NUM_CHANNELS: int = 64
    ACTION_SIZE: int = 4672  # Fixed canonical action space
    INPUT_CHANNELS: int = 32  # Board state representation
    
    # ==================== SFT Hyperparameters ====================
    sft_lr: float = 1e-3
    sft_batch_size: int = 256
    sft_epochs: int = 20
    weight_decay: float = 1e-4
    
    # ==================== RL Hyperparameters ====================
    rl_lr: float = 5e-5
    rl_batch_size: int = 64
    rl_iterations: int = 100
    games_per_iter: int = 30
    rl_num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # ==================== Data Processing ====================
    max_games: int = 8000
    pgn_paths: List[str] = field(default_factory=list)
    
    # ==================== Paths ====================
    base_path: str = field(default_factory=lambda: os.getenv('CHESS_BASE_PATH', './models'))
    sft_checkpoint_path: Optional[str] = None
    
    # ==================== Evaluation ====================
    eval_games: int = 100
    eval_interval: int = 5  # Evaluate every N iterations
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.ACTION_SIZE == 4672, f"ACTION_SIZE must be 4672, got {self.ACTION_SIZE}"
        assert self.INPUT_CHANNELS == 32, f"INPUT_CHANNELS must be 32, got {self.INPUT_CHANNELS}"
        assert self.NUM_RES_BLOCKS > 0, "NUM_RES_BLOCKS must be positive"
        assert self.NUM_CHANNELS > 0, "NUM_CHANNELS must be positive"
        assert self.sft_lr > 0, "sft_lr must be positive"
        assert self.rl_lr > 0, "rl_lr must be positive"
        print(" Configuration validated successfully")
