"""AlphaZero 式的 MCTS。

用法：
    python -m src.search.mcts --fen "..." --simulations 800

Phase 1 的所有設計都已經為這裡準備好了：盤面編碼是 canonical orientation、
著法編碼是 4672 維、policy head 的輸出可以直接當 prior、value head 當葉節點評估。

--------------------------------------------------------------------------
視角（整個檔案最容易出錯的地方，每個函式的 docstring 都會說明）
--------------------------------------------------------------------------

**約定：每個 Node 的 Q 與 value_sum，都是從「該節點輪到走棋的那一方」的視角。**

由此推出兩件事，兩件都很容易寫反：

  1. **選擇子節點時要用 `-child.q()`**。子節點的 Q 是從子節點走棋方（也就是對手）
     的視角，父節點看到的好壞要取負號。搞反的話 AI 會積極送子，因為它以為
     對手變好是好事。
  2. **回溯時每往上一層 value 取一次負號**。雙人零和，我的 +1 就是你的 -1。

`tests/test_mcts.py` 有專門的視角測試把這兩件事釘死。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from src.config import PROJECT_ROOT, Config, add_common_args, load_config
from src.encoding import encode_board, legal_indices
from src.model import ChessNet, resolve_device
from src.search import Searcher

# 終局的 value（從「輪到走棋方」的視角）：被將死就是 -1，和局是 0
TERMINAL_LOSS = -1.0
TERMINAL_DRAW = 0.0
# virtual loss 每次記一個單位的「假裝輸了」
VIRTUAL_LOSS = 1.0
# 時間管理：保留這麼多毫秒的安全餘裕，避免超時被判負
TIME_SAFETY_MARGIN_MS = 100
# 每幾次模擬檢查一次時間
TIME_CHECK_INTERVAL = 64


@dataclass
class Node:
    """MCTS 的一個節點。

    **不存 `chess.Board`**：那會吃掉大量記憶體（每個 Board 幾百 bytes，
    幾十萬個節點就是幾百 MB）。改成搜尋時沿路徑 push/pop，回溯時還原。

    Attributes:
        prior: P，父節點展開時由 policy 給出。
        visit_count: N。
        value_sum: W，**從本節點走棋方的視角**累加。
        children: {著法 index(0..4671): 子節點}。
        moves: {著法 index: chess.Move}，本節點盤面的合法著法對照表。

    `moves` 是展開時順手記下來的（`_evaluate_batch` 本來就要算一次 legal mask）。
    不記的話，選擇階段每經過一個節點就要重算一次 `legal_indices()`，
    而那是純 Python 的著法產生 + 4672 維編碼，是整個搜尋最貴的操作
    —— 實測佔掉九成以上的時間。位置固定，對照表就固定，算一次就夠了。
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    moves: dict[int, chess.Move] = field(default_factory=dict)

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    def q(self) -> float:
        """平均價值 Q = W / N，**從本節點走棋方的視角**。

        還沒被造訪過的節點回傳 0（樂觀初始化，讓它有機會被選到）。
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def puct_score(parent_visits: int, child: Node, c_puct: float) -> float:
    """算子節點的 PUCT 分數（越高越優先探索）。

        PUCT(a) = Q(a) + c_puct * P(a) * sqrt(ΣN) / (1 + N(a))

    **視角**：`child.q()` 是從**子節點走棋方**（對手）的視角，
    父節點要的是自己的視角，所以取負號。這是 MCTS 最常見的錯誤來源。

    Args:
        parent_visits: 父節點的造訪次數 ΣN。
        child: 子節點。
        c_puct: 探索係數，越大越傾向探索沒走過的著法。

    Returns:
        PUCT 分數（父節點視角）。
    """
    exploit = -child.q()          # ← 取負號，換成父節點的視角
    explore = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return exploit + explore


def terminal_value(board: chess.Board) -> float | None:
    """如果盤面已結束，回傳**當前走棋方視角**的 value。

    要涵蓋 stalemate、五十步、三次重複、子力不足，不能只判 checkmate。

    Returns:
        被將死 -1、和局 0；還沒結束則回傳 None。
    """
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return TERMINAL_DRAW
    # 有勝負時，輪到走棋的那一方一定是被將死的那方（沒有合法著法可走）
    return TERMINAL_LOSS


class MCTSSearcher(Searcher):
    """AlphaZero 式 MCTS，介面與 `GreedySearcher` 相同。

    這裡開 class 的理由跟 GreedySearcher 一樣：要記住模型、裝置與各項參數。
    另外還要在多次 `select_move` 之間保留設定（例如自我對弈時要加 Dirichlet noise）。
    """

    def __init__(
        self,
        model: ChessNet,
        device: torch.device,
        cfg: Config,
        simulations: int | None = None,
        c_puct: float | None = None,
        add_noise: bool = False,
        temperature: float = 0.0,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            model: 訓練好的網路。
            device: 推論裝置。
            cfg: 設定，讀 `cfg.mcts` 區塊。
            simulations: 每步跑幾次模擬，None 表示讀 config。
            c_puct: PUCT 的探索係數，None 表示讀 config。
            add_noise: 是否在根節點加 Dirichlet noise。
                **只有自我對弈要開**，正式對局要關（會變弱）。
            temperature: 0 = 取訪問次數最多的著法；>0 = 依 N^(1/τ) 抽樣。
            batch_size: 一次收集幾個葉節點一起送進 GPU（見 §6.4）。
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg

        mcts_cfg = cfg.mcts or {}
        self.simulations = simulations if simulations is not None else mcts_cfg.get("simulations", 800)
        self.c_puct = c_puct if c_puct is not None else mcts_cfg.get("c_puct", 2.0)
        self.dirichlet_alpha = mcts_cfg.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = mcts_cfg.get("dirichlet_epsilon", 0.25)
        self.add_noise = add_noise
        self.temperature = temperature
        self.batch_size = batch_size

        self.rng = np.random.default_rng(cfg.seed)
        # 最近一次搜尋的統計，給 UCI 的 info 行與 MoveInfo 用
        self.last_root: Node | None = None
        self.last_index_to_move: dict[int, chess.Move] = {}

    # --- 網路推論 ------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_batch(
        self, boards: list[chess.Board]
    ) -> list[tuple[dict[int, float], float, dict[int, chess.Move]]]:
        """一次前向評估多個盤面。

        單執行緒 MCTS 每次模擬只做 batch=1 的前向，GPU 使用率會掉到個位數百分比。
        批次化是讓 800 次模擬從幾十秒降到一兩秒的關鍵（規格 §6.4）。

        Args:
            boards: 要評估的盤面。

        Returns:
            每個盤面一組 (priors, value, mapping)：
              priors: {著法 index: 機率}，已套 legal mask 並正規化。
              value: **該盤面走棋方視角**的評估，-1 ~ 1。
              mapping: {著法 index: chess.Move}，順手回傳給 `Node.moves` 快取用。
        """
        if not boards:
            return []

        batch = np.stack([encode_board(b) for b in boards])
        tensor = torch.from_numpy(batch).to(self.device)
        policy_logits, values = self.model(tensor)
        policy_logits = policy_logits.float().cpu().numpy()
        values = values.squeeze(-1).float().cpu().numpy()

        results: list[tuple[dict[int, float], float, dict[int, chess.Move]]] = []
        for i, board in enumerate(boards):
            mapping = legal_indices(board)
            if not mapping:
                results.append(({}, float(values[i]), mapping))
                continue
            indices = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
            logits = policy_logits[i][indices]
            logits = logits - logits.max()          # 減最大值避免 exp 溢位
            probs = np.exp(logits)
            probs /= probs.sum()
            results.append(
                (
                    {int(idx): float(p) for idx, p in zip(indices, probs)},
                    float(values[i]),
                    mapping,
                )
            )
        return results

    # --- 一次模擬的四個步驟 --------------------------------------------------

    def _select_leaf(
        self, root: Node, board: chess.Board
    ) -> tuple[list[Node], list[int], float | None]:
        """從根節點沿 PUCT 最高的路徑往下，直到遇到未展開的節點或終局。

        **會就地修改 `board`**（一路 push），呼叫端負責 pop 回來。

        Args:
            root: 根節點。
            board: 根節點對應的盤面（會被 push 到葉節點）。

        Returns:
            (路徑上的節點, 走過的著法 index, 終局 value)
            終局 value 是**葉節點走棋方視角**；非終局時為 None。
        """
        path = [root]
        moves: list[int] = []
        node = root

        while node.is_expanded:
            best_index, best_child, best_score = -1, None, -float("inf")
            for index, child in node.children.items():
                score = puct_score(max(node.visit_count, 1), child, self.c_puct)
                if score > best_score:
                    best_index, best_child, best_score = index, child, score
            if best_child is None:
                break

            # 每個節點的對照表是展開時記下來的（見 Node.moves 的說明）。
            # 千萬不要用根節點那份 —— 同一個 index 在不同盤面代表不同的著法。
            move = node.moves.get(best_index)
            if move is None:
                break                    # 理論上不會發生，保險起見中止

            board.push(move)
            moves.append(best_index)
            path.append(best_child)
            node = best_child

        return path, moves, terminal_value(board)

    @staticmethod
    def _apply_virtual_loss(path: list[Node]) -> None:
        """沿路徑加上 virtual loss。

        目的是讓同一批的其他路徑不要全部擠到同一條線上：先假裝這條路輸了，
        它的 Q 會下降，PUCT 分數跟著下降（規格 §6.4）。
        """
        for node in path:
            node.visit_count += 1
            node.value_sum -= VIRTUAL_LOSS

    @staticmethod
    def _revert_virtual_loss(path: list[Node]) -> None:
        """把 virtual loss 扣回來（真正回溯之前一定要做）。"""
        for node in path:
            node.visit_count -= 1
            node.value_sum += VIRTUAL_LOSS

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        """沿路徑往上更新 N 與 W。

        **每往上一層 value 取一次負號**：雙人零和，葉節點走棋方的 +1
        對它的父節點（對手）就是 -1。

        Args:
            path: 從根到葉的節點串。
            value: 葉節點的評估，**葉節點走棋方視角**。
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _expand(
        self, node: Node, priors: dict[int, float], mapping: dict[int, chess.Move]
    ) -> None:
        """用 policy 的先驗機率建立子節點，並記下這個盤面的著法對照表。

        Args:
            node: 要展開的節點。
            priors: {著法 index: 先驗機率}。
            mapping: {著法 index: chess.Move}，存進 `node.moves` 供選擇階段查表。
        """
        node.moves = mapping
        for index, prior in priors.items():
            node.children[index] = Node(prior=prior)

    def _add_dirichlet_noise(self, root: Node) -> None:
        """在根節點的 prior 上加 Dirichlet noise。

            P(a) ← (1 - ε) * P(a) + ε * η(a),  η ~ Dir(α)

        沒有這個噪音，自我對弈會迅速收斂到同一條路線，資料多樣性歸零。
        **只在自我對弈時加，正式對局要關掉**（規格 §6.3）。
        """
        if not root.children:
            return
        indices = list(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(indices))
        for index, eta in zip(indices, noise):
            child = root.children[index]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(eta)

    # --- 主搜尋迴圈 ----------------------------------------------------------

    def run_simulations(
        self, board: chess.Board, simulations: int | None = None, deadline: float | None = None
    ) -> Node:
        """跑完指定次數的模擬，回傳根節點。

        Args:
            board: 要搜尋的盤面（不會被修改，內部用 copy）。
            simulations: 模擬次數，None 表示用 self.simulations。
            deadline: `time.perf_counter()` 的時間上限，超過就提早收工。

        Returns:
            根節點，`children` 的 visit_count 就是搜尋結果。
        """
        total = simulations if simulations is not None else self.simulations
        work_board = board.copy(stack=False)
        root = Node()

        # 先展開根節點
        priors, _, mapping = self._evaluate_batch([work_board])[0]
        self._expand(root, priors, mapping)
        # 根節點的對照表另外留一份，`visit_counts()` 等函式要用
        self.last_index_to_move = mapping
        if self.add_noise:
            self._add_dirichlet_noise(root)

        completed = 0
        while completed < total:
            pending: list[tuple[list[Node], int]] = []   # (path, 要 pop 幾步)
            leaf_boards: list[chess.Board] = []

            # 1. 一次收集 batch_size 條路徑到葉節點
            batch_target = min(self.batch_size, total - completed)
            for _ in range(batch_target):
                path, moves, terminal = self._select_leaf(root, work_board)

                if terminal is not None:
                    # 終局不呼叫網路，直接用 -1 / 0 回溯
                    self._backup(path, terminal)
                    completed += 1
                else:
                    # 先記 virtual loss，避免這批的其他路徑全擠到同一條
                    self._apply_virtual_loss(path)
                    pending.append((path, len(moves)))
                    leaf_boards.append(work_board.copy(stack=False))

                for _ in range(len(moves)):
                    work_board.pop()

                if terminal is None and len(leaf_boards) >= batch_target:
                    break

            # 2. 一次前向評估所有葉節點
            if leaf_boards:
                evaluations = self._evaluate_batch(leaf_boards)
                for (path, _), (priors, value, mapping) in zip(pending, evaluations):
                    self._revert_virtual_loss(path)
                    self._expand(path[-1], priors, mapping)
                    self._backup(path, value)
                    completed += 1
            elif not pending:
                break     # 全部都是終局，再跑也不會有新資訊

            if deadline is not None and time.perf_counter() >= deadline:
                break

        self.last_root = root
        return root

    # --- Searcher 介面 -------------------------------------------------------

    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """回傳訪問次數分佈 π(a) ∝ N(a)^(1/τ)。

        **這就是自我對弈的 policy target。** 注意它跟 policy head 的輸出意義不同：
        policy 是「直覺想走哪裡」，訪問次數是「想過之後認為哪裡值得」。

        Args:
            board: 原始盤面。

        Returns:
            {著法: 機率}，總和為 1。
        """
        root = self.run_simulations(board)
        mapping = root.moves

        visits = {
            mapping[index]: child.visit_count
            for index, child in root.children.items()
            if index in mapping
        }
        total = sum(visits.values())
        if total == 0:
            # 一次模擬都沒跑成（例如已經是終局），退回均勻分佈
            moves = list(board.legal_moves)
            return {m: 1.0 / len(moves) for m in moves} if moves else {}

        if self.temperature <= 0:
            best = max(visits, key=visits.get)
            return {m: (1.0 if m == best else 0.0) for m in visits}

        counts = np.array([visits[m] for m in visits], dtype=np.float64)
        powered = counts ** (1.0 / self.temperature)
        powered /= powered.sum()
        return {m: float(p) for m, p in zip(visits, powered)}

    def select_move(self, board: chess.Board) -> chess.Move:
        """搜尋後選一步棋。

        temperature = 0 時取訪問次數最多的著法；> 0 時依 N^(1/τ) 抽樣。
        """
        # 判斷依據是「有沒有合法著法」，不是 `is_game_over(claim_draw=True)`：
        # 可宣告的和棋（三次重複、五十步）盤面上還有棋可走，該走就走，
        # 要不要判和是 GUI 的事。搞錯這點會讓引擎在對打中無故棄權。
        if not any(board.legal_moves):
            raise ValueError(f"盤面已結束（{board.result()}），沒有著法可選")

        probs = self.move_probabilities(board)
        if not probs:
            raise ValueError("找不到合法著法")

        if self.temperature <= 0:
            return max(probs, key=probs.get)

        moves = list(probs)
        weights = np.array([probs[m] for m in moves], dtype=np.float64)
        weights /= weights.sum()
        return moves[int(self.rng.choice(len(moves), p=weights))]

    def visit_counts(self, board: chess.Board) -> dict[str, int]:
        """回傳 {uci: 訪問次數}，給 `MoveInfo.visits` 與 PGN 註解用。

        必須在 `select_move` / `move_probabilities` 之後呼叫（用上一次的搜尋結果）。
        """
        if self.last_root is None:
            return {}
        mapping = self.last_root.moves
        return {
            mapping[index].uci(): child.visit_count
            for index, child in self.last_root.children.items()
            if index in mapping
        }

    def principal_variation(self, board: chess.Board, max_depth: int = 8) -> list[chess.Move]:
        """從根節點沿最高訪問次數走下去，得到主要變化（UCI 的 pv）。"""
        if self.last_root is None:
            return []
        pv: list[chess.Move] = []
        node = self.last_root
        work = board.copy(stack=False)
        for _ in range(max_depth):
            if not node.children:
                break
            mapping = node.moves
            candidates = {i: c for i, c in node.children.items() if i in mapping}
            if not candidates:
                break
            best_index = max(candidates, key=lambda i: candidates[i].visit_count)
            if candidates[best_index].visit_count == 0:
                break
            move = mapping[best_index]
            pv.append(move)
            work.push(move)
            node = candidates[best_index]
        return pv


def allocate_time_ms(
    remaining_ms: int, increment_ms: int, divisor: int = 30
) -> int:
    """算這一手可以用多少毫秒（規格 §6.5）。

        本手可用時間 = 剩餘時間 / 30 + 增秒 * 0.8

    Args:
        remaining_ms: 我方剩餘時間。
        increment_ms: 每步增秒。
        divisor: 把剩餘時間分成幾手來用。

    Returns:
        可用毫秒數，至少 1（並保留 TIME_SAFETY_MARGIN_MS 的安全餘裕）。
    """
    budget = remaining_ms / divisor + increment_ms * 0.8
    budget = min(budget, max(remaining_ms - TIME_SAFETY_MARGIN_MS, 1))
    return max(int(budget), 1)


def main() -> None:
    """`python -m src.search.mcts` 對單一盤面跑一次搜尋，印出訪問次數分佈。"""
    parser = argparse.ArgumentParser(description="對單一盤面跑 MCTS 搜尋")
    add_common_args(parser)
    parser.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="要搜尋的盤面")
    parser.add_argument(
        "--checkpoint", type=str, default="models/best.pt", help="模型 checkpoint"
    )
    parser.add_argument("--simulations", type=int, default=800, help="模擬次數")
    parser.add_argument("--c-puct", type=float, default=None, help="PUCT 探索係數")
    parser.add_argument("--batch-size", type=int, default=32, help="葉節點批次大小")
    parser.add_argument("--top", type=int, default=8, help="印出前幾名")
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)

    from pathlib import Path

    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    model, _ = ChessNet.from_checkpoint(path, device=device)

    searcher = MCTSSearcher(
        model, device, cfg,
        simulations=args.simulations,
        c_puct=args.c_puct,
        batch_size=args.batch_size,
    )

    board = chess.Board(args.fen)
    print(f"盤面      : {board.fen()}")
    print(f"輪到      : {'白方' if board.turn == chess.WHITE else '黑方'}")
    print(f"模擬次數  : {args.simulations}（batch {args.batch_size}）")

    start = time.perf_counter()
    root = searcher.run_simulations(board)
    elapsed = time.perf_counter() - start

    mapping = legal_indices(board)
    rows = [
        (mapping[i], c.visit_count, c.prior, -c.q())
        for i, c in root.children.items()
        if i in mapping
    ]
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"耗時      : {elapsed:.2f}s（{args.simulations / max(elapsed, 1e-9):,.0f} 次模擬/秒）")
    print(f"\n{'著法':<8}{'訪問次數':>10}{'佔比':>8}{'prior':>9}{'Q(父視角)':>12}")
    total = sum(r[1] for r in rows) or 1
    for move, visits, prior, q in rows[: args.top]:
        print(
            f"{board.san(move):<8}{visits:>10}{visits / total * 100:>7.1f}%"
            f"{prior * 100:>8.1f}%{q:>12.3f}"
        )

    pv = searcher.principal_variation(board)
    if pv:
        work = board.copy(stack=False)
        sans = []
        for move in pv:
            sans.append(work.san(move))
            work.push(move)
        print(f"\n主要變化  : {' '.join(sans)}")


if __name__ == "__main__":
    main()
