from config import *
from parse_game import ChessConverter

class MCTSNode:
    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, device, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, fpu_mode='zero'):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.converter = ChessConverter()
        # Root Dirichlet noise params
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # First Play Urgency: 'zero' or 'parent'
        self.fpu_mode = fpu_mode

    def search(self, board, num_simulations=10, temperature=1.0, prev_board=None, root_noise=True):
        root = MCTSNode()
        search_board = board.copy()
        initial_prev_board = prev_board

        # Expand root once and add Dirichlet noise to priors
        search_board.set_fen(board.fen())
        root_value = self._expand_and_evaluate(root, search_board, initial_prev_board)
        if root_noise and len(root.children) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for (move, child), n in zip(root.children.items(), noise):
                child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * n

        for _ in range(num_simulations):
            node = root
            search_board.set_fen(board.fen())
            prev_board = initial_prev_board

            while not node.is_leaf() and not search_board.is_game_over():
                node = self._select_child(node, search_board)
                if node.move:
                    # track previous board before applying move
                    prev_for_next = search_board.copy()
                    search_board.push(node.move)
                    prev_board = prev_for_next

            if not search_board.is_game_over():
                value = self._expand_and_evaluate(node, search_board, prev_board)
            else:
                # Terminal value from the perspective of the side to move
                result = search_board.result()
                if result == "1-0":
                    value = -1.0 if search_board.turn == chess.BLACK else 1.0
                elif result == "0-1":
                    value = -1.0 if search_board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0

            self._backpropagate(node, value)

        return self._get_action_probs(root, board, temperature)

    def _select_child(self, node, board):
        best_score = -float('inf')
        best_child = None

        for move, child in node.children.items():
            if move not in board.legal_moves:
                continue
            # FPU handling: if unvisited, use parent value or zero
            if child.visit_count == 0:
                q_value = node.value() if self.fpu_mode == 'parent' else 0.0
            else:
                q_value = - child.value()
            u_value = self.c_puct * child.prior * np.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child if best_child else node

    def _expand_and_evaluate(self, node, board, prev_board):
        state = self.converter.board_to_tensor(board, prev_board=prev_board).reshape(1, 32, 8, 8)
        state_t = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(state_t)

        #raw probability từ toàn bộ 4672 đầu ra
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()

        #Normalize chỉ trên các nước đi hợp lệ
        legal_moves = list(board.legal_moves)
        move_probs = []
        total_prob = 0.0

        for move in legal_moves:
            move_idx = self.converter.encode_move(move.uci())
            if move_idx is not None:
                prob = policy[move_idx] # Lấy prob của nước đi này
            else:
                prob = 0.0
            move_probs.append(prob)
            total_prob += prob

        # Gán Prior đã chuẩn hóa vào Node con
        for move, prob in zip(legal_moves, move_probs):
            if total_prob > 0:
                prior = prob / total_prob 
            else:
                prior = 1.0 / len(legal_moves) # Fallback nếu model output toàn 0

            node.children[move] = MCTSNode(parent=node, move=move, prior=prior)

        return value

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _get_action_probs(self, root, board, temperature):
        action_probs = np.zeros(4672, dtype=np.float32)
        visits = []
        moves = []

        for move, child in root.children.items():
            if move in board.legal_moves:
                visits.append(child.visit_count)
                moves.append(move)

        if len(visits) == 0:
            return action_probs

        visits = np.array(visits, dtype=np.float32)
        if temperature == 0:
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        for move, prob in zip(moves, probs):
            move_idx = self.converter.encode_move(move.uci())
            if move_idx is not None:
                action_probs[move_idx] = prob

        return action_probs

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, samples):
        self.buffer.extend(samples)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"💾 Buffer saved: {len(self.buffer)} samples")

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data, maxlen=self.buffer.maxlen)
            print(f"📖 Buffer loaded: {len(self.buffer)} samples")
            return True
        return False