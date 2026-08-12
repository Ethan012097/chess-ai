"""MCTS 的測試（規格 §6.6）。

最重要的是**視角測試**。MCTS 的 Q 值視角搞反時，程式不會報錯、搜尋也照跑，
但 AI 會積極送子（因為它以為對手變好是好事）。這種 bug 只能靠測試抓。

全部用**隨機權重的小模型**在 CPU 上跑：
搜尋本身的正確性跟棋力無關，一步將死是靠終局偵測找到的，不是靠網路。
"""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from src.config import load_config
from src.encoding import legal_indices
from src.model import ChessNet
from src.search.mcts import (
    MCTSSearcher,
    Node,
    allocate_time_ms,
    puct_score,
    terminal_value,
)

DEVICE = torch.device("cpu")

# 白方 Ra8# 一步將死
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# 白方不吃 d5 的黑后就會被吃掉自己的后
HANGING_QUEEN_FEN = "4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1"
# 黑方已被將死
CHECKMATED_FEN = "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


@pytest.fixture(scope="module")
def searcher() -> MCTSSearcher:
    """隨機權重的小模型 + MCTS。"""
    torch.manual_seed(0)
    cfg = load_config(preset="small")
    model = ChessNet(channels=16, blocks=2)
    return MCTSSearcher(model, DEVICE, cfg, simulations=100, batch_size=8)


# --- 規格 §6.6 的三條 -------------------------------------------------------


def test_finds_mate_in_one(searcher: MCTSSearcher) -> None:
    """一步將死的盤面，100 次模擬必須找到那一步。

    這條不依賴網路品質：將死是靠終局偵測給 -1（對手視角），
    回溯之後那一步的 Q 會變成 +1，PUCT 一定會把訪問次數集中過去。
    """
    board = chess.Board(MATE_IN_ONE_FEN)
    move = searcher.select_move(board)
    assert move.uci() == "a1a8", f"沒找到一步將死，走了 {board.san(move)}"

    # 而且它應該是訪問次數最多的
    visits = searcher.visit_counts(board)
    assert max(visits, key=visits.get) == "a1a8"


def test_finds_winning_capture(searcher: MCTSSearcher) -> None:
    """不吃就輸后的盤面，搜尋後要選出吃后那一步。"""
    board = chess.Board(HANGING_QUEEN_FEN)
    searcher.simulations = 300
    try:
        move = searcher.select_move(board)
    finally:
        searcher.simulations = 100
    assert move.uci() == "d1d5", f"沒吃掉對方的后，走了 {board.san(move)}"


def test_root_q_sign_matches_network_value(searcher: MCTSSearcher) -> None:
    """視角測試：root 的 `-Q(best_child)` 應與網路直接輸出的 value 同號。

    子節點的 Q 是從**子節點走棋方**（對手）的視角，
    父節點看到的好壞要取負號。這兩者若不同號，代表某處少取或多取了一次負號。
    """
    board = chess.Board(MATE_IN_ONE_FEN)
    searcher.select_move(board)
    root = searcher.last_root
    assert root is not None

    best = max(root.children.values(), key=lambda c: c.visit_count)
    root_value = -best.q()          # ← 換成根節點（我方）的視角

    # 一步將死的盤面，我方視角一定是大好
    assert root_value > 0, f"一步將死的局面，root 視角的 Q 卻是 {root_value}"


def test_backup_alternates_sign() -> None:
    """回溯時每往上一層要取一次負號（雙人零和）。"""
    root = Node()
    child = Node()
    grandchild = Node()
    path = [root, child, grandchild]

    MCTSSearcher._backup(path, 1.0)

    assert grandchild.value_sum == pytest.approx(1.0)    # 葉節點視角
    assert child.value_sum == pytest.approx(-1.0)        # 上一層是對手
    assert root.value_sum == pytest.approx(1.0)          # 再上一層又換回來
    assert all(n.visit_count == 1 for n in path)


# --- 終局處理 ---------------------------------------------------------------


def test_terminal_value_covers_all_draw_types() -> None:
    """終局偵測要涵蓋將死、逼和，不能只判 checkmate。"""
    assert terminal_value(chess.Board(CHECKMATED_FEN)) == -1.0
    assert terminal_value(chess.Board(STALEMATE_FEN)) == 0.0
    assert terminal_value(chess.Board()) is None
    # 子力不足（只剩兩王）也是和局
    assert terminal_value(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == 0.0


def test_search_on_terminal_position_raises(searcher: MCTSSearcher) -> None:
    """已經結束的盤面沒有著法可選，要給清楚的錯誤而不是崩潰。"""
    with pytest.raises(ValueError, match="已結束"):
        searcher.select_move(chess.Board(CHECKMATED_FEN))


# --- PUCT ------------------------------------------------------------------


def test_puct_prefers_unvisited_with_high_prior() -> None:
    """沒走過但 prior 高的著法，PUCT 分數要比較高（鼓勵探索）。"""
    high = Node(prior=0.5)
    low = Node(prior=0.01)
    assert puct_score(100, high, c_puct=2.0) > puct_score(100, low, c_puct=2.0)


def test_puct_uses_negated_child_q() -> None:
    """PUCT 的利用項要用 -Q(child)。

    子節點 Q = +0.9 代表「對手很爽」，對父節點來說是壞事，分數應該被拉低。
    """
    good_for_opponent = Node(prior=0.1, visit_count=10, value_sum=9.0)   # Q = +0.9
    bad_for_opponent = Node(prior=0.1, visit_count=10, value_sum=-9.0)   # Q = -0.9
    assert puct_score(100, bad_for_opponent, 2.0) > puct_score(100, good_for_opponent, 2.0)


def test_node_q_is_zero_when_unvisited() -> None:
    """沒造訪過的節點 Q 視為 0，避免除以零。"""
    assert Node().q() == 0.0


# --- 搜尋行為 ---------------------------------------------------------------


def test_visit_counts_sum_to_simulations(searcher: MCTSSearcher) -> None:
    """根節點所有子節點的訪問次數，加起來應該接近模擬次數。"""
    board = chess.Board()
    root = searcher.run_simulations(board, simulations=64)
    total = sum(c.visit_count for c in root.children.values())
    # 批次化與 virtual loss 會讓數字有些許出入，抓寬鬆一點
    assert 32 <= total <= 96, f"訪問次數總和 {total} 不合理"


def test_move_probabilities_sum_to_one(searcher: MCTSSearcher) -> None:
    """訪問次數分佈要正規化成機率。"""
    probs = searcher.move_probabilities(chess.Board())
    assert probs
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(m in chess.Board().legal_moves for m in probs)


def test_temperature_zero_is_deterministic(searcher: MCTSSearcher) -> None:
    """temperature=0 時只會有一個著法拿到機率 1。"""
    probs = searcher.move_probabilities(chess.Board())
    assert sum(1 for p in probs.values() if p > 0) == 1


def test_selected_move_is_always_legal(searcher: MCTSSearcher) -> None:
    """各種盤面選出來的著法都必須合法。"""
    import random

    rng = random.Random(7)
    for _ in range(6):
        board = chess.Board()
        for _ in range(rng.randint(1, 30)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        assert searcher.select_move(board) in board.legal_moves


def test_dirichlet_noise_changes_priors() -> None:
    """加了 Dirichlet noise 之後根節點的 prior 應該不同（自我對弈用）。"""
    cfg = load_config(preset="small")
    model = ChessNet(channels=16, blocks=2)
    board = chess.Board()

    plain = MCTSSearcher(model, DEVICE, cfg, simulations=16, add_noise=False, batch_size=8)
    noisy = MCTSSearcher(model, DEVICE, cfg, simulations=16, add_noise=True, batch_size=8)

    root_plain = plain.run_simulations(board, simulations=16)
    root_noisy = noisy.run_simulations(board, simulations=16)

    priors_plain = {i: c.prior for i, c in root_plain.children.items()}
    priors_noisy = {i: c.prior for i, c in root_noisy.children.items()}
    assert priors_plain.keys() == priors_noisy.keys()
    assert any(
        abs(priors_plain[i] - priors_noisy[i]) > 1e-6 for i in priors_plain
    ), "加了 Dirichlet noise 但 prior 完全沒變"


def test_principal_variation_is_legal(searcher: MCTSSearcher) -> None:
    """主要變化必須是一串連續合法的著法。"""
    board = chess.Board()
    searcher.select_move(board)
    pv = searcher.principal_variation(board)
    assert pv

    work = board.copy(stack=False)
    for move in pv:
        assert move in work.legal_moves, f"pv 裡有非法著法 {move}"
        work.push(move)


def test_visit_counts_returns_uci_keys(searcher: MCTSSearcher) -> None:
    """visit_counts 回傳的 key 是 UCI 字串（給 MoveInfo.visits 用）。"""
    board = chess.Board()
    searcher.select_move(board)
    visits = searcher.visit_counts(board)
    assert visits
    for uci, count in visits.items():
        assert chess.Move.from_uci(uci) in board.legal_moves
        assert count >= 0


# --- 時間管理（§6.5）--------------------------------------------------------


def test_allocate_time_formula() -> None:
    """本手可用時間 = 剩餘時間 / 30 + 增秒 * 0.8。"""
    assert allocate_time_ms(60000, 1000) == pytest.approx(60000 / 30 + 800, abs=1)
    assert allocate_time_ms(30000, 0) == pytest.approx(1000, abs=1)


def test_allocate_time_never_exceeds_remaining() -> None:
    """時間快用完時不能配置超過剩餘時間，要留安全餘裕。"""
    assert allocate_time_ms(200, 0) < 200
    assert allocate_time_ms(50, 0) >= 1        # 再怎麼緊也要回傳正數


def test_deadline_stops_search_early(searcher: MCTSSearcher) -> None:
    """給一個已經過期的 deadline，搜尋要立刻收工而不是跑滿。"""
    import time

    board = chess.Board()
    root = searcher.run_simulations(
        board, simulations=100000, deadline=time.perf_counter() + 0.05
    )
    total = sum(c.visit_count for c in root.children.values())
    assert total < 100000, "deadline 沒有生效"
