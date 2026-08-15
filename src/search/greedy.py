"""Phase 1 的「搜尋」：policy 取最高分的合法著法，加上兩層很便宜的補強。

流程：
  1. 編碼盤面（含鏡射）→ 前向傳播 → policy logits
  2. 用 legal mask 把非法著法設成 -inf
  3. softmax
  4. 鏡射還原：原本輪到黑方時，把 index 轉回來的 move 鏡射回原盤面座標
  5. temperature=0 取 argmax，>0 則依機率抽樣

兩層補強（CPU 也負擔得起，但明顯提升棋力）：
  - **一步將死檢查**：先掃所有合法著法，有立即將死就直接走
  - **送子檢查**：對 policy 前 k 名的著法各走一步，用 value head 評估對手視角的
    分數，選對自己最好的那個（等同 depth-2 的極小化極大）

為什麼送子檢查有用：policy head 學的是「人類會走什麼」，不是「什麼棋好」。
它有時候會走出一步看起來很自然、但白白送掉一隻子的棋。用 value head 檢查一層
就能擋掉大部分這種情況。
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from src.config import Config
from src.encoding import NUM_INPUT_PLANES, encode_board, legal_indices
from src.model import ChessNet
from src.search import Searcher, find_mate_in_one

# 對手已經被將死時，value head 不會被呼叫，直接給最大分
MATE_SCORE = 1.0


class GreedySearcher(Searcher):
    """用 policy head 直接下棋的 searcher。"""

    def __init__(
        self,
        model: ChessNet,
        device: torch.device,
        cfg: Config,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> None:
        """
        Args:
            model: 訓練好的網路（會被切到 eval 模式）。
            device: 推論裝置。
            cfg: 設定（讀 search 區塊）。
            temperature: 覆寫 cfg.search.temperature。0 = argmax。
            top_k: 覆寫 cfg.search.top_k（送子檢查要看幾個候選）。
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg
        self.temperature = cfg.search.temperature if temperature is None else temperature
        self.top_k = cfg.search.top_k if top_k is None else top_k
        self.use_mate_check = cfg.search.use_mate_check
        self.use_lookahead = cfg.search.use_lookahead

    # --- 網路推論 ------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """對單一盤面做一次前向傳播。

        Args:
            board: 原始盤面。

        Returns:
            (policy_logits, value)
              policy_logits: (4672,) 的 numpy 陣列（canonical 視角，未 mask）
              value: 純量，**當前走棋方**的視角，+1 代表當前走棋方會贏
        """
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        return policy_logits[0].float().cpu().numpy(), float(value.item())

    @torch.no_grad()
    def _evaluate_batch(self, boards: list[chess.Board]) -> np.ndarray:
        """一次評估多個盤面的 value（送子檢查用，比一個一個算快很多）。

        Args:
            boards: 盤面清單。

        Returns:
            (N,) 的 value，每個都是**該盤面走棋方**的視角。
        """
        if not boards:
            return np.zeros(0, dtype=np.float32)
        batch = np.stack([encode_board(b) for b in boards])
        x = torch.from_numpy(batch).to(self.device)
        _, value = self.model(x)
        return value.squeeze(-1).float().cpu().numpy()

    # --- 機率分佈 ------------------------------------------------------------

    def analyse(self, board: chess.Board) -> tuple[float, dict[chess.Move, float]]:
        """一次前向傳播同時取得 value 與 policy 機率。

        `move_probabilities` 只要機率，但 CLI 顯示、PGN 註解、網頁 API 都同時需要
        value，分開呼叫等於白跑兩次前向傳播，所以拆一支公用的出來。

        Args:
            board: 原始盤面。

        Returns:
            (value, {原始座標的著法: 機率})
              value: **當前走棋方視角**，-1 ~ 1。
              機率: 已套 legal mask 與 softmax、已鏡射還原，總和為 1。
              盤面已結束時回傳 (value, {})。
        """
        mapping = legal_indices(board)   # {canonical index: 原始座標的 move}
        policy_logits, value = self._evaluate(board)
        if not mapping:
            return value, {}

        indices = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
        moves = list(mapping.values())

        # 只取合法著法的 logits（等同把其餘設成 -inf 再 softmax）
        legal_logits = policy_logits[indices]
        legal_logits = legal_logits - legal_logits.max()   # 減最大值避免 exp 溢位
        probs = np.exp(legal_logits)
        probs /= probs.sum()

        return value, {move: float(p) for move, p in zip(moves, probs)}

    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """算出每個合法著法的機率（已套 legal mask 與 softmax，已鏡射還原）。

        Args:
            board: 原始盤面。

        Returns:
            {原始座標的著法: 機率}，總和為 1。盤面已結束時回傳空 dict。
        """
        return self.analyse(board)[1]

    # --- 補強 ---------------------------------------------------------------

    def _lookahead_best(
        self, board: chess.Board, candidates: list[tuple[chess.Move, float]]
    ) -> chess.Move:
        """送子檢查：對候選著法各走一步，用 value head 挑對自己最好的。

        走完一步之後輪到對手，value head 給的是**對手視角**的分數，
        所以要取負號才是我方的分數（極小化極大的 depth-2）。

        Args:
            board: 原始盤面。
            candidates: [(著法, policy 機率)]，已依機率由高到低排序。

        Returns:
            最好的著法。
        """
        moves = [m for m, _ in candidates]
        scores: list[float] = []
        to_evaluate: list[chess.Board] = []
        eval_slots: list[int] = []

        for i, move in enumerate(moves):
            board.push(move)
            if board.is_checkmate():
                scores.append(MATE_SCORE)            # 直接將死，最高分
            elif board.is_stalemate() or board.is_insufficient_material():
                scores.append(0.0)                   # 和局
            else:
                scores.append(float("nan"))          # 待會用網路批次評估
                to_evaluate.append(board.copy(stack=False))
                eval_slots.append(i)
            board.pop()

        if to_evaluate:
            # 對手視角的 value，取負號變成我方視角
            opponent_values = self._evaluate_batch(to_evaluate)
            for slot, v in zip(eval_slots, opponent_values):
                scores[slot] = -float(v)

        best_idx = int(np.argmax(scores))
        return moves[best_idx]

    # --- 主入口 -------------------------------------------------------------

    def select_move(self, board: chess.Board) -> chess.Move:
        """選一步棋。

        Args:
            board: 原始盤面。

        Returns:
            合法著法。

        Raises:
            ValueError: 盤面已經結束（沒有合法著法）。
        """
        # 同 mcts.py：判斷依據是「有沒有合法著法」。子力不足、五十步這類
        # 「和棋但還有棋可走」的盤面要照常走，判和是 GUI 的事。
        if not any(board.legal_moves):
            raise ValueError(f"盤面已結束（{board.result()}），沒有著法可選")

        # 1. 一步將死檢查（最便宜也最有效）
        if self.use_mate_check:
            mate_move = find_mate_in_one(board)
            if mate_move is not None:
                return mate_move

        probs = self.move_probabilities(board)
        if not probs:
            raise ValueError("找不到合法著法")

        # 2. temperature > 0：依機率抽樣（用來製造變化，評估時通常設 0）
        if self.temperature > 0:
            moves = list(probs.keys())
            weights = np.array(list(probs.values()), dtype=np.float64)
            # temperature 越大越平均；用 p^(1/T) 重新正規化
            weights = weights ** (1.0 / self.temperature)
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                return max(probs, key=probs.get)
            weights /= total
            return moves[int(np.random.choice(len(moves), p=weights))]

        # 3. 送子檢查：對 policy 前 k 名做 depth-2 極小化極大
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if self.use_lookahead and self.top_k > 1 and len(ranked) > 1:
            return self._lookahead_best(board, ranked[: self.top_k])

        # 4. 純 argmax
        return ranked[0][0]


class RandomSearcher(Searcher):
    """隨機走合法著法。`evaluate.py --mode baseline` 的對手，也是最低門檻。"""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        """隨機挑一個合法著法。"""
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("盤面已結束，沒有著法可選")
        return moves[int(self.rng.integers(len(moves)))]

    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """所有合法著法機率相同。"""
        moves = list(board.legal_moves)
        if not moves:
            return {}
        p = 1.0 / len(moves)
        return {m: p for m in moves}
