"""
Validate core infrastructure - Test dimension consistency
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.models.resnet import SmallResNet
from core.chess_logic.move_encoding import CanonicalMoveEncoder
from core.chess_logic.board_encoding import BoardEncoder
import torch
import chess


def test_config():
    """Test configuration loading and validation"""
    print("\n" + "="*60)
    print("TEST 1: Configuration")
    print("="*60)
    
    config = Config()
    config.validate()
    
    print(f" Config validated")
    print(f"   ACTION_SIZE: {config.ACTION_SIZE}")
    print(f"   NUM_RES_BLOCKS: {config.NUM_RES_BLOCKS}")
    print(f"   NUM_CHANNELS: {config.NUM_CHANNELS}")


def test_move_encoder():
    """Test canonical move encoding"""
    print("\n" + "="*60)
    print("TEST 2: Move Encoding")
    print("="*60)
    
    encoder = CanonicalMoveEncoder()
    
    # Test encoding
    test_moves = ['e2e4', 'e7e5', 'g1f3', 'e7e8q', 'e7e8r']
    for move_uci in test_moves:
        idx = encoder.encode_move(move_uci)
        if idx is not None:
            decoded = encoder.decode_move(idx)
            print(f" {move_uci} -> {idx} -> {decoded}")
        else:
            print(f"  {move_uci} -> None (not in canonical space)")
    
    print(f"\n Action space size: {encoder.action_size}")


def test_board_encoder():
    """Test board state encoding"""
    print("\n" + "="*60)
    print("TEST 3: Board Encoding")
    print("="*60)
    
    encoder = BoardEncoder()
    board = chess.Board()
    
    # Encode initial position
    tensor = encoder.encode(board)
    
    print(f" Board encoded to tensor")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Non-zero channels: {(tensor.sum(axis=(1,2)) > 0).sum()}")


def test_model():
    """Test model architecture"""
    print("\n" + "="*60)
    print("TEST 4: Model Architecture")
    print("="*60)
    
    config = Config()
    model = SmallResNet(config)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 32, 8, 8)
    
    policy_logits, value = model(dummy_input)
    
    print(f" Model forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Policy logits shape: {policy_logits.shape}")
    print(f"   Value shape: {value.shape}")
    
    # Validate dimensions
    assert policy_logits.shape == (batch_size, 4672), f"Policy shape mismatch: {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Value shape mismatch: {value.shape}"
    
    arch_info = model.get_architecture_info()
    print(f"\n Architecture info:")
    for key, val in arch_info.items():
        print(f"   {key}: {val}")


def test_dimension_consistency():
    """Test SFT and RL dimension consistency"""
    print("\n" + "="*60)
    print("TEST 5: Dimension Consistency (SFT ↔ RL)")
    print("="*60)
    
    config = Config()
    
    # Create two models (simulating SFT and RL)
    sft_model = SmallResNet(config)
    rl_model = SmallResNet(config)
    
    # Check dimensions match
    sft_policy_dim = sft_model.policy_fc.out_features
    rl_policy_dim = rl_model.policy_fc.out_features
    
    assert sft_policy_dim == rl_policy_dim == 4672, \
        f"Dimension mismatch: SFT={sft_policy_dim}, RL={rl_policy_dim}"
    
    print(f" Dimension consistency verified")
    print(f"   SFT policy output: {sft_policy_dim}")
    print(f"   RL policy output: {rl_policy_dim}")
    print(f"    Both models use identical 4672 action space")


def main():
    print("\n" + "="*60)
    print("CORE INFRASTRUCTURE VALIDATION")
    print("="*60)
    
    try:
        test_config()
        test_move_encoder()
        test_board_encoder()
        test_model()
        test_dimension_consistency()
        
        print("\n" + "="*60)
        print(" ALL TESTS PASSED")
        print("="*60)
        print("\nCore infrastructure is ready for SFT and RL training!")
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
