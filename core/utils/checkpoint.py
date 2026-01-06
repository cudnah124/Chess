"""
Checkpoint Management - Save/load with dimension validation
"""
import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    metadata: Dict[str, Any]
):
    """
    Save model checkpoint with dimension metadata for validation
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        path: Save path
        metadata: Additional metadata (epoch, loss, stage, etc.)
    """
    # Extract model architecture info
    if hasattr(model, 'get_architecture_info'):
        arch_info = model.get_architecture_info()
    else:
        # Fallback: extract from model structure
        arch_info = {
            'action_size': model.policy_fc.out_features if hasattr(model, 'policy_fc') else None,
            'num_channels': model.conv_input.out_channels if hasattr(model, 'conv_input') else None,
        }
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'architecture': arch_info,
        'metadata': metadata
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")
    print(f"   Action size: {arch_info.get('action_size', 'unknown')}")
    print(f"   Metadata: {metadata}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    expected_action_size: int = 4672,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load checkpoint with automatic dimension validation
    
    Args:
        model: PyTorch model to load weights into
        path: Checkpoint path
        optimizer: Optional optimizer to load state into
        strict: Whether to strictly enforce state dict matching
        expected_action_size: Expected action size (default 4672)
        device: Device to load checkpoint to
    
    Returns:
        metadata: Dictionary containing training info
    
    Raises:
        ValueError: If action_size mismatch detected
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Handle both wrapped and raw checkpoints
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        architecture = checkpoint.get('architecture', {})
        metadata = checkpoint.get('metadata', {})
    else:
        state_dict = checkpoint
        architecture = {}
        metadata = {}
    
    # Validate action size
    if 'action_size' in architecture:
        actual_action_size = architecture['action_size']
        if actual_action_size != expected_action_size:
            raise ValueError(
                f" Action size mismatch!\n"
                f"   Expected: {expected_action_size}\n"
                f"   Checkpoint: {actual_action_size}\n"
                f"   This checkpoint is incompatible with the current model.\n"
                f"   Please retrain with action_size={expected_action_size}"
            )
    
    # Load state dict
    model.load_state_dict(state_dict, strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f" Warning: Could not load optimizer state: {e}")
    
    print(f" Checkpoint loaded: {path}")
    if architecture:
        print(f"   Architecture: {architecture}")
    if metadata:
        print(f"   Metadata: {metadata}")
    
    return metadata


def validate_checkpoint_dimensions(path: str, expected_action_size: int = 4672) -> bool:
    """
    Validate checkpoint dimensions without loading into model
    
    Args:
        path: Checkpoint path
        expected_action_size: Expected action size
    
    Returns:
        True if dimensions match, False otherwise
    """
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'architecture' in checkpoint:
            arch = checkpoint['architecture']
            actual_size = arch.get('action_size', None)
            
            if actual_size is None:
                print(f" No action_size found in checkpoint")
                return False
            
            if actual_size == expected_action_size:
                print(f" Dimensions match: action_size={actual_size}")
                return True
            else:
                print(f" Dimension mismatch: expected {expected_action_size}, got {actual_size}")
                return False
        else:
            print(f" No architecture metadata in checkpoint")
            return False
            
    except Exception as e:
        print(f" Error validating checkpoint: {e}")
        return False
