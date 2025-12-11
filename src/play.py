import pygame
import chess
import sys
import numpy as np
import random
import os
import time

# --- IMPORT CÁC CLASS TỪ FILE MODEL CỦA BẠN ---
# Giả sử file chứa SmallResNet, AlphaZeroTrainer tên là model.py
# Nếu bạn để chung 1 file thì không cần dòng này
from model import * # ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
WIDTH, HEIGHT = 512, 512 # Kích thước cửa sổ (bội số của 8)
DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION
MAX_FPS = 15
IMAGES = {}

# Màu sắc bàn cờ (Light/Dark squares)
COLOR_LIGHT = (240, 217, 181) 
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT = (186, 202, 68) # Màu khi chọn ô

# ==========================================
# 2. HÀM KHỞI TẠO TRAINER (CỦA BẠN)
# ==========================================
def init_trainer():
    # Initialize
    model = SmallResNet(num_res_blocks=6, num_channels=64, action_size=4672)

    trainer = AlphaZeroTrainer(model=model, device=device, lr=0.00005, weight_decay=1e-4)

    trainer.converter = ChessConverter(move_to_idx=CANONICAL_MOVE_TO_IDX, idx_to_move=CANONICAL_IDX_TO_MOVE)
    trainer.mcts.converter = ChessConverter(move_to_idx=CANONICAL_MOVE_TO_IDX, idx_to_move=CANONICAL_IDX_TO_MOVE)

    MODEL_LOAD_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_rl_best.pth')

    # Load Model
    if os.path.exists(MODEL_LOAD_PATH):
        print(f"🔄 Loading Model: {MODEL_LOAD_PATH}")
        trainer.load_checkpoint(MODEL_LOAD_PATH)
        print("✅ Model loaded!")
    else:
        print(f"❌ Model not found: {MODEL_LOAD_PATH}")
    
    return trainer

def load_images():
    """Load ảnh quân cờ vào dictionary IMAGES"""
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        path = os.path.join(os.path.dirname(__file__), "images", piece + ".png")
        if os.path.exists(path):
            # Load và scale ảnh vừa khít ô cờ
            IMAGES[piece] = pygame.transform.scale(pygame.image.load(path), (SQ_SIZE, SQ_SIZE))
        else:
            # Tạo ô màu tạm nếu không có ảnh
            print(f"⚠️ Thiếu ảnh: {path}")
            surf = pygame.Surface((SQ_SIZE, SQ_SIZE))
            surf.fill((255, 0, 0)) # Màu đỏ báo lỗi
            IMAGES[piece] = surf

def draw_board(screen):
    """Vẽ các ô vuông bàn cờ"""
    colors = [pygame.Color(COLOR_LIGHT), pygame.Color(COLOR_DARK)]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    """Vẽ quân cờ lên bàn"""
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            # Pygame vẽ từ trên xuống (Row 0), python-chess rank 0 là dưới cùng
            # Công thức chuyển đổi:
            square_idx = chess.square(c, 7-r) 
            piece = board.piece_at(square_idx)
            if piece:
                # Lấy tên file ảnh tương ứng (vd: 'wP', 'bK')
                color_prefix = 'w' if piece.color == chess.WHITE else 'b'
                piece_name = f"{color_prefix}{piece.symbol().upper()}"
                screen.blit(IMAGES[piece_name], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_highlight(screen, selected_square):
    """Highlight ô đang chọn"""
    if selected_square is not None:
        c = chess.square_file(selected_square)
        r = 7 - chess.square_rank(selected_square)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100) # Độ trong suốt
        s.fill(pygame.Color(COLOR_HIGHLIGHT))
        screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

# ==========================================
# 4. GAME LOOP CHÍNH
# ==========================================
def main():
    # 1. Khởi tạo Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AlphaZero Chess AI")
    clock = pygame.time.Clock()
    
    # 2. Load tài nguyên
    load_images()
    
    # 3. Khởi tạo Bot
    print("⏳ Đang khởi tạo Bot...")
    trainer = init_trainer()
    board = chess.Board()
    
    # 4. Chọn phe
    # (Để đơn giản trong GUI, mình mặc định người chơi Trắng, hoặc bạn có thể input ở console trước khi cửa sổ hiện lên)
    player_color = None
    while player_color not in ['w', 'b']:
        player_color = input("Bạn muốn cầm quân nào? (w=Trắng, b=Đen): ").lower()
    player_is_white = (player_color == 'w')

    print(f"🎮 Bắt đầu! Bạn cầm {'TRẮNG' if player_is_white else 'ĐEN'}")

    # Biến trạng thái game
    selected_square = None # Ô đang chọn (Chess square index)
    running = True
    game_over = False
    
    while running:
        human_turn = (board.turn == chess.WHITE and player_is_white) or \
                     (board.turn == chess.BLACK and not player_is_white)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- XỬ LÝ CLICK CHUỘT (CHỈ KHI LƯỢT NGƯỜI) ---
            if not game_over and human_turn and event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos() # (x, y)
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                
                # Chuyển đổi tọa độ click sang index của python-chess (0-63)
                clicked_sq = chess.square(col, 7-row)
                
                if selected_square == clicked_sq:
                    # Click lại ô đã chọn -> Hủy chọn
                    selected_square = None
                else:
                    if selected_square is None:
                        # Chưa chọn gì -> Chọn quân (nếu đúng màu)
                        piece = board.piece_at(clicked_sq)
                        if piece and piece.color == board.turn:
                            selected_square = clicked_sq
                    else:
                        # Đã chọn quân -> Click ô đích -> Thử đi
                        move = chess.Move(selected_square, clicked_sq)
                        
                        # Xử lý phong cấp tự động (Auto-Queen) để đơn giản hóa GUI
                        # Nếu quân là Tốt và đi đến hàng cuối
                        p = board.piece_at(selected_square)
                        if p and p.piece_type == chess.PAWN:
                            rank = chess.square_rank(clicked_sq)
                            if rank == 0 or rank == 7:
                                move = chess.Move(selected_square, clicked_sq, promotion=chess.QUEEN)
                        
                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None # Reset sau khi đi
                        else:
                            # Nếu click vào quân khác cùng màu -> Đổi lựa chọn
                            piece = board.piece_at(clicked_sq)
                            if piece and piece.color == board.turn:
                                selected_square = clicked_sq
                            else:
                                selected_square = None # Click sai -> Hủy

        # --- VẼ GIAO DIỆN ---
        draw_board(screen)
        draw_highlight(screen, selected_square)
        draw_pieces(screen, board)
        
        # --- LOGIC BOT ---
        if not game_over and not human_turn:
            # Update màn hình để người chơi thấy nước đi của mình trước khi Bot nghĩ
            pygame.display.flip()
            
            print("🤖 Bot đang suy nghĩ...", end=" ")
            # MCTS Search
            policy = trainer.mcts.search(board, num_simulations=100, temperature=0.0, root_noise=False)
            
            # Chọn move
            legal_moves = list(board.legal_moves)
            probs = []
            for m in legal_moves:
                idx = trainer.converter.encode_move(m.uci())
                if idx is not None: probs.append(policy[idx])
                else: probs.append(0.0)
            
            if sum(probs) > 0:
                chosen = legal_moves[np.argmax(probs)]
            else:
                chosen = random.choice(legal_moves)
            
            print(f"-> {chosen}")
            board.push(chosen)

        # --- KIỂM TRA KẾT THÚC ---
        if board.is_game_over():
            draw_pieces(screen, board) # Vẽ lại lần cuối
            pygame.display.flip()
            print("🏁 GAME OVER")
            print("Kết quả:", board.result())
            game_over = True
            time.sleep(5) # Đợi 5s rồi thoát
            running = False

        pygame.display.flip()
        clock.tick(MAX_FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()