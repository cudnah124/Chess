"""
Test script to validate notebook code syntax
"""
import json

def validate_notebook(notebook_path):
    """Check if notebook cells have valid Python syntax"""
    print(f"\n[TEST] Validating: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    print(f"  Found {len(code_cells)} code cells")
    
    errors = []
    for i, cell in enumerate(code_cells, 1):
        source = ''.join(cell['source'])
        if source.strip():
            try:
                compile(source, f'<cell {i}>', 'exec')
            except SyntaxError as e:
                errors.append(f"Cell {i}: {e}")
    
    if errors:
        print(f"  [ERROR] Found {len(errors)} syntax errors:")
        for err in errors:
            print(f"    - {err}")
        return False
    else:
        print(f"  [OK] All cells have valid syntax")
        return True

if __name__ == "__main__":
    notebooks = [
        'notebooks/train_sft.ipynb',
        'notebooks/train_rl.ipynb'
    ]
    
    all_valid = True
    for nb in notebooks:
        if not validate_notebook(nb):
            all_valid = False
    
    if all_valid:
        print(f"\n[SUCCESS] All notebooks validated successfully!")
    else:
        print(f"\n[FAILED] Some notebooks have errors")
        exit(1)
