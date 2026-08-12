"""`MoveInfo`：一步棋的完整紀錄，以及 value ↔ centipawn 的換算。

這個結構被 PGN 輸出（第 1 節）、網頁棋盤（第 4 節）、MCTS（第 6 節）共用，
所以獨立成一個模組，避免那三邊互相 import。**定好就不要再改。**

--------------------------------------------------------------------------
視角約定（整個專案錯誤率最高的地方，每次出現都要講清楚）
--------------------------------------------------------------------------

- `MoveInfo.value`：**當前走棋方**的視角。+1 代表「輪到走的這一方會贏」。
  這跟 value head 的輸出、跟 Phase 1 訓練資料的 `result` 欄位都是同一個約定。
- `[%eval]`（PGN）與網頁的評估條：**白方**視角。所以黑方走棋時要取負號。
- `score cp`（UCI info）：**當前走棋方**視角，不用轉換。

換句話說，只有寫進 PGN 與網頁時要轉成白方視角，其餘一律維持走棋方視角。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import chess

# policy_top 存前 8 名而非 5 名：網頁預設畫 5 條箭頭，但要讓使用者能調整，
# 先多存幾個比較省事（規格 §1.1）。
POLICY_TOP_N = 8

# --- tanh 值 ↔ centipawn 換算（規格 §1.3）---------------------------------
# value 接近 ±1 時 tan 會爆掉，先夾住再算。
VALUE_CLAMP = 0.9999
CP_CLAMP = 10000
# Leela 的換算公式，數值感比單純線性更貼近 Stockfish
LEELA_CP_SCALE = 290.68
LEELA_CP_SLOPE = 1.548


@dataclass
class MoveInfo:
    """一步棋的完整紀錄。

    Attributes:
        move: 實際走的著法。
        san: SAN 表示，給前端顯示用。
        value: value head 輸出，**當前走棋方視角**，-1 ~ 1。
            這是「走這步之前」的盤面評估（對應 `fen_before`）。
        policy_top: [(uci, prob), ...] 前 POLICY_TOP_N 名，已套 legal mask 並正規化。
        visits: MCTS 訪問次數 {uci: N}；Phase 1 的 greedy searcher 為 None。
        elapsed_ms: 這一步想了多久。
        fen_before: 走這步**之前**的盤面。
    """

    move: chess.Move
    san: str
    value: float
    policy_top: list[tuple[str, float]] = field(default_factory=list)
    visits: dict[str, int] | None = None
    elapsed_ms: int = 0
    fen_before: str = ""

    @property
    def turn_before(self) -> chess.Color:
        """走這步的是哪一方。用 fen_before 判斷，不用另外存一個欄位。"""
        return chess.Board(self.fen_before).turn if self.fen_before else chess.WHITE

    def value_white(self) -> float:
        """把 value 轉成**白方視角**（PGN 的 [%eval] 與網頁評估條用這個）。

        Returns:
            -1 ~ 1，正值代表白方佔優。
        """
        return self.value if self.turn_before == chess.WHITE else -self.value


def value_to_cp(value: float, linear: bool = False) -> int:
    """把 tanh 空間的 value 換算成 centipawn（百分兵值）。

    只影響顯示，不影響棋力。預設用 Leela 公式，數值感比線性更貼近 Stockfish。

    Args:
        value: -1 ~ 1。**視角由呼叫端決定**——傳白方視角的值進來就得到白方視角的 cp。
        linear: True 改用簡單線性 `600 * value`。

    Returns:
        centipawn，夾在 ±CP_CLAMP。
    """
    clamped = max(-VALUE_CLAMP, min(VALUE_CLAMP, value))
    if linear:
        cp = 600.0 * clamped
    else:
        cp = LEELA_CP_SCALE * math.tan(LEELA_CP_SLOPE * clamped)
    return int(max(-CP_CLAMP, min(CP_CLAMP, cp)))


def cp_to_value(cp: float) -> float:
    """`value_to_cp` 的反函數：centipawn → tanh 空間。

    第 5 節用 Stockfish 評分重訓 value head 時會用到（把 cp 轉成訓練 target）。

    Args:
        cp: centipawn。視角由呼叫端決定。

    Returns:
        -0.99 ~ 0.99 的 value。
    """
    value = math.atan(cp / LEELA_CP_SCALE) / LEELA_CP_SLOPE
    return max(-0.99, min(0.99, value))


def make_move_info(
    board: chess.Board,
    move: chess.Move,
    value: float,
    move_probs: dict[chess.Move, float],
    elapsed_ms: int,
    visits: dict[str, int] | None = None,
) -> MoveInfo:
    """組出一個 `MoveInfo`。

    刻意不吃 `Searcher` 而是吃已經算好的數值，這樣 greedy 與 MCTS 都能用同一支，
    也不會造成模組之間互相 import。

    Args:
        board: 走這步**之前**的盤面（函式不會修改它）。
        move: 要走的著法。
        value: value head 對 `board` 的評估，**當前走棋方視角**。
        move_probs: {著法: 機率}，`Searcher.move_probabilities` 的輸出。
        elapsed_ms: 這一步花了多少毫秒。
        visits: MCTS 訪問次數；greedy 傳 None。

    Returns:
        MoveInfo。
    """
    ranked = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)
    policy_top = [(m.uci(), float(p)) for m, p in ranked[:POLICY_TOP_N]]
    return MoveInfo(
        move=move,
        san=board.san(move),
        value=float(value),
        policy_top=policy_top,
        visits=visits,
        elapsed_ms=int(elapsed_ms),
        fen_before=board.fen(),
    )


def main() -> None:
    """`python -m src.move_info` 印出 value ↔ centipawn 的對照表。

    用來直觀感受 Leela 公式在 value 接近 ±1 時放大得多厲害
    —— 這解釋了為什麼評分曲線看起來比 Stockfish 誇張。
    """
    parser = argparse.ArgumentParser(description="value ↔ centipawn 換算對照表")
    parser.add_argument(
        "--linear", action="store_true", help="改用線性公式 600 * value"
    )
    parser.add_argument(
        "--value", type=float, default=None, help="只換算單一個 value"
    )
    args = parser.parse_args()

    if args.value is not None:
        cp = value_to_cp(args.value, linear=args.linear)
        print(f"value {args.value:+.4f} → {cp:+d} cp（{cp / 100:+.2f} 兵）")
        return

    formula = "600 * value（線性）" if args.linear else "290.68 * tan(1.548 * value)（Leela）"
    print(f"換算公式：{formula}\n")
    print(f"{'value':>8}{'centipawn':>12}{'兵值':>10}")
    for value in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
        cp = value_to_cp(value, linear=args.linear)
        print(f"{value:>8.2f}{cp:>12d}{cp / 100:>10.2f}")
    print("\n（負的 value 完全對稱，符號相反）")


if __name__ == "__main__":
    main()
