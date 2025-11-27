Phase I (Static Learning): The model is initialized via Imitation Learning (Supervised Training) on high-quality human PGN data. This creates a stable baseline policy before entering the dynamic RL phase.

Phase II (Dynamic RL): The engine enters a Self-Play loop, where the network's policy is continuously refined using MCTS-generated data.

Key Features
Architecture: Optimized Small ResNet backbone (5-6 blocks) for memory efficiency.

Input: Custom 32-channel tensor capturing T-1 move history and auxiliary game state (castling rights, repetition count).

Search: MCTS (No costly random rollouts; uses Value Head for rapid evaluation).

Project Structure
This structure separates the core logic (src/) from configuration, data, and model checkpoints.

```
Deep_Chess_Project/
├── data/
│   ├── raw_pgn/              # Raw PGN files (Human Games)
│   └── processed/            # Serialized Tensor data (.npz, .h5)
│
├── src/
│   ├── config.py             # Hyperparameters (Batch size, LR, ResBlocks)
│   ├── model.py              # SmallResNet Class (PyTorch)
│   ├── parse_game.py         # Data parser (PGN -> 32-Channel Tensor)
│   ├── mcts.py               # Monte Carlo Tree Search Logic
│   ├── train_supervised.py   # Code for Phase I (Imitation Learning)
│   ├── train_rl.py           # Code for Phase II (RL Self-Play Loop)
│   └── play.py               # Game Interface / Engine Testing
│
├── models/                   # Model Checkpoints
│   ├── model_supervised.pth  # Baseline model from Phase I
│   └── model_rl.pth          # Best performing RL model
│
└── requirements.txt          # Dependencies: torch, python-chess, numpy
```
