import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import chess
import os
import pickle
from collections import deque
import random
from tqdm import tqdm

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
if device.type == 'cpu':
    print("⚠️ Warning: Đang chạy trên CPU. Hãy bật GPU T4 trong Runtime > Change runtime type")

# =========================
# Global Config (easy to edit)
# =========================
MODEL_PATH = '../models/'            # Path to read existing models/resources
SAVE_PATH = '../models/'            # Path to save new checkpoints/buffers/logs
MODEL_NAME_BEST = 'model_rl_best.pth'
SFT_MODEL_NAME = 'model_sft.pth'
BUFFER_NAME_LATEST = 'buffer_latest.pkl'
BUFFER_NAME_RL_LATEST = 'buffer_rl_latest.pkl'
TRAINING_LOG_NAME = 'training_log.csv'

# Derived absolute paths
MODEL_BEST_PATH = os.path.join(SAVE_PATH, MODEL_NAME_BEST)
SFT_MODEL_PATH = os.path.join(MODEL_PATH, SFT_MODEL_NAME)
BUFFER_LATEST_PATH = os.path.join(SAVE_PATH, BUFFER_NAME_LATEST)
BUFFER_RL_LATEST_PATH = os.path.join(SAVE_PATH, BUFFER_NAME_RL_LATEST)
TRAINING_LOG_PATH = os.path.join(SAVE_PATH, TRAINING_LOG_NAME)
MODEL_LATEST_PATH = os.path.join(SAVE_PATH, 'model_rl_latest.pth')
MOVE_MAP_LATEST_PATH = os.path.join(SAVE_PATH, 'move_map.pkl')

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# =========================
# Canonical 4672 Move Map (Hard-coded AlphaZero-style)
# =========================
def build_canonical_move_map_4672():
    move_to_idx = {}
    idx_to_move = {}
    idx = 0

    # 1. Queen Moves (8 hướng * 7 ô) = 3584 indices
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for from_sq in range(64):
        for d_r, d_f in directions:
            for dist in range(1, 8):
                to_rank = (from_sq // 8) + d_r * dist
                to_file = (from_sq % 8) + d_f * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    dest = to_rank * 8 + to_file
                    uci = chess.Move(from_sq, dest).uci()
                    move_to_idx[uci] = idx
                    idx_to_move[idx] = uci
                idx += 1

    # 2. Knight Moves (8 hướng) = 512 indices
    knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    for from_sq in range(64):
        for d_r, d_f in knight_moves:
            to_rank = (from_sq // 8) + d_r
            to_file = (from_sq % 8) + d_f
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                dest = to_rank * 8 + to_file
                uci = chess.Move(from_sq, dest).uci()
                move_to_idx[uci] = idx
                idx_to_move[idx] = uci
            idx += 1

    # 3. Underpromotions (3 hướng * 3 loại quân) = 576 indices
    file_steps = [0, -1, 1]
    promotions = ['r', 'b', 'n']

    for from_sq in range(64):
        rank = from_sq // 8
        file = from_sq % 8

        # Xác định hướng phong cấp dựa trên vị trí quân tốt
        rank_step = 0
        if rank == 6:
            rank_step = 1  # Trắng: Lên
        elif rank == 1:
            rank_step = -1 # Đen: Xuống

        for f_step in file_steps:
            for p in promotions:
                if rank_step != 0:
                    to_rank = rank + rank_step
                    to_file = file + f_step
                    if 0 <= to_file < 8:
                        dest = to_rank * 8 + to_file
                        uci = chess.Move(from_sq, dest, promotion=chess.Piece.from_symbol(p).piece_type).uci()
                        move_to_idx[uci] = idx
                        idx_to_move[idx] = uci
                idx += 1
    return move_to_idx, idx_to_move

# Khởi tạo lại
CANONICAL_MOVE_TO_IDX, CANONICAL_IDX_TO_MOVE = build_canonical_move_map_4672()
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_res_blocks=6, num_channels=64, action_size=4672):
        super(SmallResNet, self).__init__()
        # Input: 32 channels (Current + History + Aux)
        self.conv_input = nn.Conv2d(32, num_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Backbone: Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy Head (Actor)
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value Head (Critic)
        self.value_conv = nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy: Softmax
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 8 * 8)
        policy_logits = self.policy_fc(p)  # Trả về RAW LOGITS

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 3 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return policy_logits, v