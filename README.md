# Chess AI - AlphaZero Implementation

A deep reinforcement learning chess engine implementing the AlphaZero algorithm with supervised fine-tuning and Monte Carlo Tree Search self-play.

---

## Overview

This project implements a complete chess AI training pipeline inspired by DeepMind's AlphaZero. The system uses a two-stage approach: supervised learning from expert games followed by reinforcement learning through self-play with MCTS.

**Key Features:**
- ResNet-based policy-value network (~10M parameters)
- Canonical 4672-action move encoding for all legal chess moves
- MCTS-guided self-play for reinforcement learning
- Flexible training via CLI or Google Colab notebooks
- Comprehensive evaluation tools

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Training Your Model

**Stage 1 - Supervised Fine-Tuning:**
```bash
python -m sft.train configs/sft_config.yaml
```

**Stage 2 - Reinforcement Learning:**
```bash
python -m rl.train configs/rl_config.yaml
```

**Alternative - Google Colab:**
- Upload `notebooks/train_sft.ipynb` or `notebooks/train_rl.ipynb`
- All dependencies are self-contained
- No additional setup required

### Evaluation

```bash
# Test against Minimax AI
python tests/test_vs_minimax.py models/your_model.pth --games 20 --depth 3

# Watch games in detail
python tests/test_vs_minimax.py models/your_model.pth --games 2 --verbose

# Evaluate performance
python scripts/evaluate.py models/your_model.pth --games 100
```

---

## Architecture

**Neural Network:**
- Input: 32-channel 8×8 board representation
- Backbone: 6-block ResNet with 64 channels
- Policy Head: 4672-dimensional move probabilities
- Value Head: Scalar position evaluation (-1 to +1)

**Training Pipeline:**
1. **SFT Stage:** Learn from PGN game databases
2. **RL Stage:** Improve through MCTS self-play
3. **Evaluation:** Test against Minimax or other models

---

## Project Structure

```
ML/
├── core/              # Core components
│   ├── models/       # Neural architectures
│   ├── chess_logic/  # Move/board encoding
│   └── utils/        # Checkpoints, logging
├── sft/              # Supervised training
├── rl/               # Reinforcement learning
│   ├── mcts/        # Monte Carlo Tree Search
│   └── self_play/   # Self-play generation
├── configs/          # Training configurations
├── notebooks/        # Colab-ready notebooks
├── scripts/          # Utilities
├── tests/            # Evaluation tools
└── models/           # Checkpoints
```

---

## Configuration

Customize training in `configs/sft_config.yaml` or `configs/rl_config.yaml`:

**Model Architecture:**
- ResNet blocks and channels
- Input/output dimensions (ACTION_SIZE=4672 is fixed)

**Training Parameters:**
- Learning rate, batch size, epochs
- MCTS simulations, temperature
- Data paths and checkpoint locations

---

## Training Data

Place PGN files in `data/` folder. Recommended sources:
- [FICS Games Database](https://www.ficsgames.org/download.html) - Large collection of rated games
- [Lichess Database](https://database.lichess.org/) - Monthly game archives
- [CCRL](https://www.computerchess.org.uk/ccrl/) - Computer chess games

---

## Performance Benchmarks

Training times (CPU, approximate):
- **SFT:** 2-5 min/epoch (500 games)
- **RL:** 10-20 min/iteration (30 games, 100 MCTS simulations)

GPU acceleration significantly reduces training time.

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- python-chess
- NumPy, tqdm, PyYAML

See `requirements.txt` for complete dependencies.

---

## Testing Your Model

Simply place your `.pth` checkpoint in `models/` and run:

```bash
python tests/test_vs_minimax.py models/your_model.pth
```

The test automatically validates model compatibility and reports win rates.

---

## License

Educational use only.

## Acknowledgments

Based on the AlphaZero algorithm by DeepMind (Silver et al., 2017).
