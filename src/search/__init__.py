"""搜尋策略的共同介面。

Phase 1 只有 `GreedySearcher`（policy 取最高分的合法著法 + 一點便宜的補強），
Phase 2 會加上 `MCTSSearcher`。兩者共用同一個介面，所以 `play.py` 與
`evaluate.py` 不需要為了換搜尋方式而改寫。

這裡開 class 的理由：searcher 需要「記住」模型與裝置（每次下棋都要用到），
用函式的話每次呼叫都要把 model / device / config 全部再傳一次。
"""

from __future__ import annotations

import chess


class Searcher:
    """選擇著法的抽象介面。

    子類別必須實作 `select_move` 與 `move_probabilities`。
    刻意不用 abc.ABC，保持簡單；沒實作就會在呼叫時得到清楚的 NotImplementedError。
    """

    def select_move(self, board: chess.Board) -> chess.Move:
        """選一步棋。

        Args:
            board: 目前盤面（原始座標系，不是 canonical）。

        Returns:
            合法著法。
        """
        raise NotImplementedError("子類別必須實作 select_move")

    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """回傳每個合法著法的機率。

        Args:
            board: 目前盤面。

        Returns:
            {著法: 機率}，總和為 1。Phase 2 的 MCTS 會回傳訪問次數分佈。
        """
        raise NotImplementedError("子類別必須實作 move_probabilities")


__all__ = ["Searcher"]
