"""
Dimension Validation Script - Check checkpoint compatibility
"""
import sys
import os
import torch
from pathlib import Path


def validate_checkpoint(checkpoint_path: str, expected_action_size: int = 4672):
    """
    Validate checkpoint dimensions
    
    Args:
        checkpoint_path: Path to checkpoint file
        expected_action_size: Expected action size (default 4672)
    """
    print(f"\n{'='*60}")
    print(f"Validating: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print(f" File not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            architecture = checkpoint.get('architecture', {})
            metadata = checkpoint.get('metadata', {})
            
            print(f" Checkpoint format: Wrapped")
            
            # Display metadata
            if metadata:
                print(f"\n📊 Metadata:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
            
            # Check architecture
            if architecture:
                print(f"\n Architecture:")
                for key, value in architecture.items():
                    print(f"   {key}: {value}")
                
                # Validate action size
                actual_action_size = architecture.get('action_size', None)
                if actual_action_size is not None:
                    if actual_action_size == expected_action_size:
                        print(f"\n VALID: action_size={actual_action_size} (matches expected)")
                        return True
                    else:
                        print(f"\n INVALID: action_size={actual_action_size} (expected {expected_action_size})")
                        return False
                else:
                    print(f"\n WARNING: No action_size in architecture metadata")
            else:
                print(f"\n WARNING: No architecture metadata found")
        else:
            # Raw state dict
            state_dict = checkpoint
            print(f" Checkpoint format: Raw state_dict")
            
            # Try to infer action size from policy layer
            if 'policy_fc.weight' in state_dict:
                actual_action_size = state_dict['policy_fc.weight'].shape[0]
                print(f"\n🔍 Inferred action_size from policy_fc: {actual_action_size}")
                
                if actual_action_size == expected_action_size:
                    print(f" VALID: action_size={actual_action_size}")
                    return True
                else:
                    print(f" INVALID: action_size={actual_action_size} (expected {expected_action_size})")
                    return False
            else:
                print(f"\n WARNING: Cannot determine action_size")
        
        return False
        
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        return False


def validate_directory(directory: str, expected_action_size: int = 4672):
    """Validate all .pth files in directory"""
    print(f"\n{'='*60}")
    print(f"Validating directory: {directory}")
    print(f"{'='*60}")
    
    pth_files = list(Path(directory).glob('**/*.pth'))
    
    if not pth_files:
        print(f" No .pth files found in {directory}")
        return
    
    print(f"\nFound {len(pth_files)} checkpoint(s)")
    
    results = []
    for pth_file in pth_files:
        is_valid = validate_checkpoint(str(pth_file), expected_action_size)
        results.append((pth_file.name, is_valid))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    valid_count = sum(1 for _, is_valid in results if is_valid)
    
    for filename, is_valid in results:
        status = " VALID" if is_valid else " INVALID"
        print(f"{status}: {filename}")
    
    print(f"\nTotal: {valid_count}/{len(results)} valid")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/validate_dimensions.py <checkpoint_path>")
        print("  python scripts/validate_dimensions.py --dir <directory>")
        print("\nExample:")
        print("  python scripts/validate_dimensions.py models/sft_best.pth")
        print("  python scripts/validate_dimensions.py --dir models/")
        sys.exit(1)
    
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 3:
            print(" Error: --dir requires a directory path")
            sys.exit(1)
        validate_directory(sys.argv[2])
    else:
        checkpoint_path = sys.argv[1]
        is_valid = validate_checkpoint(checkpoint_path)
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
