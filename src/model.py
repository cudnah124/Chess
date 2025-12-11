from mcts import *
    
class AlphaZeroTrainer:
    def __init__(self, model, device=device, lr=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.converter = ChessConverter()
        self.use_self_play = False
        self.mcts = MCTS(model=self.model, device=device, c_puct=1.0)

    def combined_loss(self, policy_pred, value_pred, policy_target, value_target):
        # Policy loss: cross-entropy with log-softmax
        log_policy = torch.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_target * log_policy).sum(dim=1).mean()
        value_loss = nn.MSELoss()(value_pred, value_target)
        return policy_loss + value_loss, policy_loss, value_loss

    def _get_ai_policy(self, board, prev_board=None):
        state = self.converter.board_to_tensor(board, prev_board=prev_board).reshape(1, 32, 8, 8)
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(state_t)
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy_probs

    def train_on_batch(self, batch):
        self.model.train()
        states, policies, values = zip(*batch)
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        policies = torch.stack([torch.FloatTensor(p) for p in policies]).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device)

        policy_pred, value_pred = self.model(states)
        loss, policy_loss, value_loss = self.combined_loss(policy_pred, value_pred, policies, values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'use_self_play': self.use_self_play,
        }, path)
        print(f"💾 Saved: {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        # Detect policy head size from checkpoint
        policy_key_w = 'policy_fc.weight'
        policy_key_b = 'policy_fc.bias'
        if policy_key_w in state_dict:
            target_action_size = state_dict[policy_key_w].shape[0]
            current_action_size = self.model.policy_fc.out_features
            if target_action_size != current_action_size:
                print(f"⚙️ Rebuilding model: checkpoint outputs={target_action_size}, current={current_action_size}")
                # Recreate model with the target action size
                new_model = SmallResNet(num_res_blocks=6, num_channels=64, action_size=target_action_size).to(self.device)
                # Replace trainer model and MCTS model reference
                self.model = new_model
                if hasattr(self, 'mcts') and self.mcts is not None:
                    self.mcts.model = self.model
                # Recreate optimizer tied to new parameters
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'], weight_decay=self.optimizer.param_groups[0].get('weight_decay', 1e-4))
                # Recreate scheduler
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        # Load weights
        self.model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️ Optimizer state mismatch; reinitialized. Details: {e}")
        # Restore scheduler state if present
        if isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️ Scheduler state mismatch; reinitialized. Details: {e}")
        if isinstance(checkpoint, dict) and 'use_self_play' in checkpoint:
            self.use_self_play = checkpoint['use_self_play']
        print(f"📖 Loaded: {path}")
    
# AlphaZeroTrainer - Game playing methods
def self_play_game(self, num_simulations=10, temperature=1.0):
    board = chess.Board()
    samples = []
    move_count = 0
    ai_plays_white = random.choice([True, False])
    prev_board = None

    def _safe_sample_move(legal_moves, probs_like):
        # Convert and sanitize
        probs = np.array(probs_like, dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip negatives and renormalize
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s <= 0.0:
            return random.choice(legal_moves)
        probs /= s
        # Adjust last element to fix any floating drift
        if len(probs) > 0:
            drift = 1.0 - probs.sum()
            probs[-1] += drift
            # Final safety clip and renorm
            probs = np.clip(probs, 0.0, None)
            s = probs.sum()
            if s <= 0.0:
                return random.choice(legal_moves)
            probs /= s
        try:
            idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[idx]
        except ValueError:
            # Fallback to uniform if numpy rejects p
            return random.choice(legal_moves)

    while not board.is_game_over() and move_count < 200:
        state = self.converter.board_to_tensor(board, prev_board=prev_board)
        current_turn_is_white = board.turn == chess.WHITE

        if self.use_self_play:
            temp = 1.0 if move_count < 30 else 0.0
            policy_target = self.mcts.search(board, num_simulations=num_simulations, temperature=temp, prev_board=prev_board, root_noise=True)

            legal_moves = list(board.legal_moves)
            legal_probs = [policy_target[self.converter.encode_move(m.uci())] if self.converter.encode_move(m.uci()) is not None else 0.0 for m in legal_moves]

            if sum(legal_probs) > 0:
                selected_move = _safe_sample_move(legal_moves, legal_probs)
            else:
                selected_move = random.choice(legal_moves)

            samples.append((state, policy_target, None, current_turn_is_white))
            prev_for_next = board.copy()
            board.push(selected_move)
            prev_board = prev_for_next
        else:
            if (board.turn == chess.WHITE) == ai_plays_white:
                policy = self._get_ai_policy(board, prev_board=prev_board)
                legal_moves = list(board.legal_moves)
                legal_probs = [policy[self.converter.encode_move(m.uci())] if self.converter.encode_move(m.uci()) is not None else 0.0 for m in legal_moves]

                if sum(legal_probs) > 0:
                    selected_move = _safe_sample_move(legal_moves, legal_probs)
                else:
                    selected_move = random.choice(legal_moves)

                policy_target = np.zeros(4672, dtype=np.float32)
                move_idx = self.converter.encode_move(selected_move.uci())
                if move_idx is not None:
                    policy_target[move_idx] = 1.0

                samples.append((state, policy_target, None, current_turn_is_white))
                prev_for_next = board.copy()
                board.push(selected_move)
                prev_board = prev_for_next
            else:
                prev_for_next = board.copy()
                board.push(random.choice(list(board.legal_moves)))
                prev_board = prev_for_next

        move_count += 1

    result = board.result()
    white_outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else -0.1)

    final_samples = []
    for state, policy, _, was_white_turn in samples:
        value = white_outcome if was_white_turn else -white_outcome
        final_samples.append((state, policy, value))

    return final_samples, white_outcome, ai_plays_white if not self.use_self_play else None

AlphaZeroTrainer.self_play_game = self_play_game



# AlphaZeroTrainer -  Evaluation & Training methods
import csv

# Preload SFT teacher model once
SFT_TEACHER = None
SFT_PATH = SFT_MODEL_PATH
if os.path.exists(SFT_PATH):
    SFT_TEACHER = SmallResNet(num_res_blocks=6, num_channels=64, action_size=4672).to(device)
    checkpoint = torch.load(SFT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        SFT_TEACHER.load_state_dict(checkpoint['model_state_dict'])
    else:
        SFT_TEACHER.load_state_dict(checkpoint)
    SFT_TEACHER.eval()
    print("👨‍🏫 SFT teacher loaded")
else:
    print("ℹ️ No SFT teacher found; Random-only opponent available")


def evaluate_vs_random(self, num_games=100):
    wins, losses, draws = 0, 0, 0
    self.model.eval()

    for _ in tqdm(range(num_games), desc="Evaluating"):
        board = chess.Board()
        ai_plays_white = random.choice([True, False])
        move_count = 0
        prev_board = None

        while not board.is_game_over() and move_count < 500:
            if (board.turn == chess.WHITE) == ai_plays_white:
                policy = self.mcts.search(board, num_simulations=50, temperature=0.0, prev_board=prev_board, root_noise=False)
                legal_moves = list(board.legal_moves)
                legal_probs = [policy[self.converter.encode_move(m.uci())] if self.converter.encode_move(m.uci()) is not None else 0.0 for m in legal_moves]

                if sum(legal_probs) > 0:
                    # Greedy for evaluation, but keep safety
                    probs = np.array(legal_probs, dtype=np.float64)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    probs = np.clip(probs, 0.0, None)
                    chosen = legal_moves[np.argmax(probs)]
                else:
                    chosen = random.choice(legal_moves)
                prev_for_next = board.copy()
                board.push(chosen)
                prev_board = prev_for_next
            else:
                prev_for_next = board.copy()
                board.push(random.choice(list(board.legal_moves)))
                prev_board = prev_for_next
            move_count += 1

        result = board.result()

        if result == "1/2-1/2":
            outcome = 0.0
        elif result == "1-0":
            outcome = 1.0 if ai_plays_white else -1.0
        else:
            outcome = -1.0 if ai_plays_white else 1.0

        if outcome > 0: wins += 1
        elif outcome < 0: losses += 1
        else: draws += 1

    winrate = (wins / num_games * 100) if num_games > 0 else 0
    return wins, losses, draws, winrate

def evaluate_vs_sft(self, sft_path=None, num_games=10):
    # 1. Load SFT Model
    if sft_path is None:
        sft_path = SFT_MODEL_PATH

    try:
        sft_opponent = SmallResNet(num_res_blocks=6, num_channels=64, action_size=4672).to(self.device)
        checkpoint = torch.load(sft_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            sft_opponent.load_state_dict(checkpoint['model_state_dict'])
        else:
            sft_opponent.load_state_dict(checkpoint)

        sft_opponent.eval() # Freeze SFT
        print("✅ SFT Opponent Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading SFT model: {e}")
        print("💡 Hãy kiểm tra lại đường dẫn file .pth")
        return

    wins, losses, draws = 0, 0, 0
    self.model.eval()

    print(f"\n⚔️  RL BOT (MCTS) vs SFT MODEL (Raw Policy) | {num_games} Games")

    for i in range(num_games):
        board = chess.Board()
        ai_plays_white = random.choice([True, False])

        # In thông tin ván đấu
        color_str = "White" if ai_plays_white else "Black"
        print(f"   Game {i+1}: RL Bot ({color_str}) vs SFT...", end=" ", flush=True)

        move_count = 0
        prev_board = None

        while not board.is_game_over() and move_count < 200:
            is_rl_turn = (board.turn == chess.WHITE) == ai_plays_white

            if is_rl_turn:
                # === RL BOT ===
                policy = self.mcts.search(board, num_simulations=50, temperature=0.0, prev_board=prev_board, root_noise=False)
                legal_moves = list(board.legal_moves)
                legal_probs = []
                for m in legal_moves:
                    idx = self.converter.encode_move(m.uci())
                    if idx is not None:
                        legal_probs.append(policy[idx])
                    else:
                        legal_probs.append(0.0)

                if sum(legal_probs) > 0:
                    chosen = legal_moves[np.argmax(legal_probs)]
                else:
                    chosen = random.choice(legal_moves)
            else:
                # === SFT OPPONENT ===
                state = self.converter.board_to_tensor(board, prev_board=prev_board).reshape(1, 32, 8, 8)
                state_t = torch.FloatTensor(state).to(self.device)

                # 2. Forward pass
                with torch.no_grad():
                    policy_logits, _ = sft_opponent(state_t)

                # 3. Lấy probabilities
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

                # 4. Masking legal moves
                legal_moves = list(board.legal_moves)
                move_probs = []
                for m in legal_moves:
                    idx = self.converter.encode_move(m.uci())
                    if idx is not None:
                        move_probs.append(policy_probs[idx])
                    else:
                        move_probs.append(0.0)

                # 5. Greedy selection
                if sum(move_probs) > 0:
                    chosen = legal_moves[np.argmax(move_probs)]
                else:
                    chosen = random.choice(legal_moves) # Fallback

            prev_for_next = board.copy()
            board.push(chosen)
            prev_board = prev_for_next
            move_count += 1

        result = board.result()
        if result == "1/2-1/2":
            draws += 1
            print("Draw 🤝")
        elif result == "1-0":
            if ai_plays_white: wins += 1; print("Win 🏆")
            else: losses += 1; print("Loss ❌")
        else: # 0-1
            if ai_plays_white: losses += 1; print("Loss ❌")
            else: wins += 1; print("Win 🏆")

    winrate = (wins / num_games * 100) if num_games > 0 else 0
    print(f"\n📊 TOTAL: Wins={wins}, Losses={losses}, Draws={draws} | Winrate: {winrate:.1f}%")
    return wins, losses, draws, winrate


def _opponent_move(self, board, opponent_model, prev_board_ref):
    if opponent_model is None:
        # Random opponent
        prev_for_next = board.copy()
        board.push(random.choice(list(board.legal_moves)))
        prev_board_ref[0] = prev_for_next
        return
    # SFT teacher move (greedy on policy)
    state = self.converter.board_to_tensor(board, prev_board=prev_board_ref[0]).reshape(1, 32, 8, 8)
    state_t = torch.FloatTensor(state).to(self.device)
    with torch.no_grad():
        policy_logits, _ = opponent_model(state_t)

    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    legal_moves = list(board.legal_moves)
    legal_probs = [policy[self.converter.encode_move(m.uci())] if self.converter.encode_move(m.uci()) is not None else 0.0 for m in legal_moves]
    if sum(legal_probs) > 0:
        probs = np.array(legal_probs, dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        chosen = legal_moves[np.argmax(probs)]
    else:
        chosen = random.choice(legal_moves)
    prev_for_next = board.copy()
    board.push(chosen)
    prev_board_ref[0] = prev_for_next


def train_iteration(self, num_games=50, batch_size=64, train_steps=500, eval_vs_random=False, iteration=0):
    # Chỉ self-play khi training
    num_sims = 100  
    print(f"\n{'='*60}\nSelf-Play: {num_games} games (MCTS {num_sims} sims)\n{'='*60}")
    self.model.eval()
    for i in tqdm(range(num_games), desc="Self-Play"):
        samples, _, _ = self.self_play_game(num_simulations=num_sims, temperature=1.0)
        self.replay_buffer.add_batch(samples)


    print(f"📊 Buffer: {len(self.replay_buffer)} samples")

    if not self.use_self_play:
        if len(self.replay_buffer) > 5000:
            print("🚀 KÍCH HOẠT CHẾ ĐỘ SELF-PLAY: Bot đã đủ mạnh!")
            self.use_self_play = True

    print(f"\n{'='*60}\nTraining\n{'='*60}")
    epochs = 5
    batches_per_epoch = max(1, len(self.replay_buffer) // batch_size)
    total_steps = batches_per_epoch * epochs
    print(f"Training: {epochs} epochs, {batches_per_epoch} batches/epoch = {total_steps} total steps")

    for step in range(total_steps):
        batch = self.replay_buffer.sample(batch_size)
        loss, p_loss, v_loss = self.train_on_batch(batch)
        if step % 100 == 0:
            epoch = step // batches_per_epoch + 1
            print(f"  Epoch {epoch}/{epochs}, Step {step}/{total_steps}: Loss={loss:.4f}, P={p_loss:.4f}, V={v_loss:.4f}")

    # Step LR scheduler after each iteration
    self.scheduler.step()

    if eval_vs_random:
        wins, losses, draws, winrate = self.evaluate_vs_random(num_games=10)
        print(f"🎯 Post-Train Eval Random: {winrate:.1f}%")

        wins, losses, draws, winrate = self.evaluate_vs_sft(sft_path='/content/drive/MyDrive/Chess/models/model_best_4672.pth', num_games=10)
        print(f"🎯 Post-Train Eval SFT: {winrate:.1f}%")
        return wins, losses, draws, winrate, len(self.replay_buffer), num_sims
    else:
        return None, None, None, None, len(self.replay_buffer), num_sims


def train(self, num_iterations=100, games_per_iter=40, save_interval=5, target_winrate=90.0, log_csv_path=None):
    print("\n" + "="*60 + "\n🚀 AlphaZero RL Training\n" + "="*60)
    best_winrate = 0.0

    # Setup CSV logging
    if log_csv_path is None:
        log_csv_path = TRAINING_LOG_PATH
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "wins", "losses", "draws", "winrate", "buffer_size", "num_sims", "lr"])

    for iteration in range(num_iterations):
        print(f"\n{'='*60}\nIteration {iteration + 1}/{num_iterations}\n{'='*60}")
        eval_vs_random = (iteration + 1) % 2 == 0
        current_lr = self.optimizer.param_groups[0]['lr']
        wins, losses, draws, winrate, buffer_size, num_sims = self.train_iteration(num_games=games_per_iter, eval_vs_random=eval_vs_random, iteration=iteration)

        # Track and save best
        if winrate and winrate > best_winrate:
            best_winrate = winrate
            self.save_checkpoint(MODEL_BEST_PATH)
            print(f"🏆 NEW BEST: {winrate:.1f}%")

        # Write CSV row if evaluated
        if winrate is not None:
            with open(log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration + 1, wins, losses, draws, winrate, buffer_size, num_sims, current_lr])

        # Always save latest (overwrite)
        self.save_checkpoint(MODEL_LATEST_PATH)
        self.replay_buffer.save(BUFFER_LATEST_PATH)
        # ✅ Also persist move_map after each iteration

        # Early stopping on target winrate
        if winrate and winrate >= target_winrate:
            print(f"\n{'='*60}\n🎉 EARLY STOP: Target {target_winrate:.1f}% reached (winrate {winrate:.1f}%)\n{'='*60}")
            break

    print(f"\n✅ Training completed! Best winrate: {best_winrate:.1f}%")

AlphaZeroTrainer.evaluate_vs_sft = evaluate_vs_sft
AlphaZeroTrainer.evaluate_vs_random = evaluate_vs_random
AlphaZeroTrainer.train_iteration = train_iteration
AlphaZeroTrainer.train = train
AlphaZeroTrainer._opponent_move = _opponent_move