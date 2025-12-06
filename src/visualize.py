import pygame
import chess
import torch
import numpy as np
import os
import sys
import pickle
import random
import requests
import time
from torch import nn
import torch.nn.functional as F

# --- CẤU HÌNH (CONFIG) ---
WINDOW_SIZE = 640
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 30
ASSET_DIR = "assets/pieces" # Thư mục chứa ảnh quân cờ

# Đường dẫn Model (Sửa lại cho đúng vị trí trên máy bạn)
MODEL_PATH = "./weights/model_best_2000.pth" 
MAP_PATH = "./weights/move_map.pkl"
DELAY_MOVE = 0.5 # Thời gian nghỉ giữa các nước đi (giây)

# Màu sắc
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT = (186, 202, 68)

# --- 1. MODEL & CONVERTER CLASSES (Copy từ dự án) ---
class ChessConverter:
    def __init__(self):
        self.piece_map = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
        self.move_to_idx = {}
        self.idx_to_move = {}
        self.next_idx = 0
    def load_moves_map(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
                self.move_to_idx = d['move_to_idx']
                self.idx_to_move = d['idx_to_move']
                self.next_idx = len(self.move_to_idx)
    def board_to_tensor(self, board, prev_board=None):
        tensor = np.zeros((32, 8, 8), dtype=np.float32)
        for sq, pc in board.piece_map().items():
            tensor[self.piece_map[pc.symbol()]][chess.square_rank(sq)][chess.square_file(sq)] = 1
        if prev_board:
            for sq, pc in prev_board.piece_map().items():
                tensor[self.piece_map[pc.symbol()]+12][chess.square_rank(sq)][chess.square_file(sq)] = 1
        if board.turn == chess.WHITE: tensor[24,:,:] = 1
        if board.has_kingside_castling_rights(chess.WHITE): tensor[25,:,:] = 1
        if board.has_queenside_castling_rights(chess.WHITE): tensor[26,:,:] = 1
        if board.has_kingside_castling_rights(chess.BLACK): tensor[27,:,:] = 1
        if board.has_queenside_castling_rights(chess.BLACK): tensor[28,:,:] = 1
        if board.ep_square:
            tensor[29][chess.square_rank(board.ep_square)][chess.square_file(board.ep_square)] = 1
        if board.is_repetition(1): tensor[30,:,:] = 1
        if board.is_repetition(2): tensor[31,:,:] = 1
        return tensor

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

class SmallResNet(nn.Module):
    def __init__(self, num_res_blocks=6, num_channels=64, action_size=4672):
        super().__init__()
        self.conv_input = nn.Conv2d(32, num_channels, 3, 1, 1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*8*8, action_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 3, 1, bias=False), nn.BatchNorm2d(3), nn.ReLU(),
            nn.Flatten(), nn.Linear(3*8*8, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )
    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks: x = block(x)
        return F.softmax(self.policy_head(x), dim=1), self.value_head(x)

# --- 2. HỖ TRỢ TẢI ẢNH (AUTO DOWNLOAD ASSETS) ---
def download_assets():
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)
    
    # URL bộ quân cờ chuẩn (Lichess style)
    base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/"
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    
    print("Checking assets...")
    for p in pieces:
        filename = os.path.join(ASSET_DIR, f"{p}.svg")
        # Pygame không hỗ trợ SVG tốt, ta dùng link PNG nếu có hoặc user tự convert. 
        # Để đơn giản, ta dùng bộ ảnh PNG từ một nguồn khác tương thích Pygame.
        png_url = f"https://images.chesscomfiles.com/chess-themes/pieces/neo/150/{p.lower()}.png"
        save_path = os.path.join(ASSET_DIR, f"{p}.png")
        
        if not os.path.exists(save_path):
            print(f"Downloading {p}.png...")
            try:
                r = requests.get(png_url)
                if r.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(r.content)
            except:
                print(f"Failed to download {p}")

# --- 3. AGENT CLASSES ---
class NeuralAgent:
    def __init__(self, model_path, map_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.converter = ChessConverter()
        self.converter.load_moves_map(map_path)
        action_size = max(self.converter.next_idx, 4672)
        self.model = SmallResNet(action_size=action_size).to(self.device)
        # Load weights
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            print("✅ Neural Agent Loaded")
        else:
            print("❌ Model not found! Playing random moves.")
            self.model = None

    def get_move(self, board):
        if self.model is None: return random.choice(list(board.legal_moves))
        
        state = self.converter.board_to_tensor(board)
        state_t = torch.tensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.model(state_t)
        
        legal_moves = list(board.legal_moves)
        policy_np = policy.cpu().numpy()[0]
        
        best_move = None
        best_prob = -1.0
        
        for move in legal_moves:
            idx = self.converter.move_to_idx.get(move.uci(), None)
            if idx is not None and idx < len(policy_np):
                if policy_np[idx] > best_prob:
                    best_prob = policy_np[idx]
                    best_move = move
        
        return best_move if best_move else random.choice(legal_moves)

class RandomAgent:
    def get_move(self, board):
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None

# --- 4. GIAO DIỆN GUI (PYGAME) ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Neural Chess AI")
    clock = pygame.time.Clock()
    
    # Tải ảnh quân cờ
    download_assets()
    pieces_imgs = {}
    piece_names = {'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
                   'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'}
    
    for char, name in piece_names.items():
        path = os.path.join(ASSET_DIR, f"{name}.png")
        if os.path.exists(path):
            img = pygame.image.load(path)
            pieces_imgs[char] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))

    # Khởi tạo Game
    board = chess.Board()
    neural = NeuralAgent(MODEL_PATH, MAP_PATH)
    random_bot = RandomAgent()
    
    running = True
    game_over = False
    
    while running:
        clock.tick(FPS)
        
        # 1. Vẽ Bàn Cờ
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(screen, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                
                # Highlight nước vừa đi
                if board.move_stack:
                    last_move = board.peek()
                    if last_move.to_square == chess.square(col, 7-row) or \
                       last_move.from_square == chess.square(col, 7-row):
                        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                        s.set_alpha(100)
                        s.fill(HIGHLIGHT)
                        screen.blit(s, (col*SQUARE_SIZE, row*SQUARE_SIZE))

        # 2. Vẽ Quân Cờ
        # FEN string loop
        board_fen = board.board_fen()
        row, col = 0, 0
        for char in board_fen:
            if char == '/':
                row += 1
                col = 0
            elif char.isdigit():
                col += int(char)
            else:
                if char in pieces_imgs:
                    screen.blit(pieces_imgs[char], (col*SQUARE_SIZE, row*SQUARE_SIZE))
                col += 1

        # 3. Xử lý Event (Thoát)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()

        # 4. Logic Game AI
        if not game_over:
            if board.is_game_over():
                print(f"GAME OVER: {board.result()}")
                game_over = True
                continue

            # AI Thinking... (Xử lý sự kiện để không bị treo màn hình)
            pygame.event.pump() 
            
            if board.turn == chess.WHITE:
                # Neural đi (Trắng)
                move = neural.get_move(board)
                board.push(move)
                print(f"Neural (White): {move}")
                time.sleep(DELAY_MOVE)
            else:
                # Random đi (Đen)
                move = random_bot.get_move(board)
                board.push(move)
                print(f"Random (Black): {move}")
                time.sleep(DELAY_MOVE)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()