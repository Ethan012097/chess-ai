"""`move_info.py` 與 `pgn_writer.py` 的測試。

重點全在**視角**：`MoveInfo.value` 是當前走棋方視角，但 PGN 的 `[%eval]` 是白方
視角。這個轉換搞反的話，貼到 lichess 的評分曲線會整條上下顛倒，而且因為兩邊
數值都「看起來很合理」，用眼睛非常難發現。

全部在 CPU 上跑，不需要模型或 GPU。
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn
import pytest

from src.move_info import (
    POLICY_TOP_N,
    MoveInfo,
    cp_to_value,
    make_move_info,
    value_to_cp,
)
from src.pgn_writer import (
    REQUIRED_HEADERS,
    build_game,
    format_eval_comment,
    format_policy_comment,
    write_game,
)


def _info(fen: str, uci: str, value: float) -> MoveInfo:
    """用指定盤面與 value 造一個 MoveInfo。"""
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    return MoveInfo(
        move=move,
        san=board.san(move),
        value=value,
        policy_top=[(uci, 0.5)],
        elapsed_ms=30,
        fen_before=board.fen(),
    )


# --- 視角轉換：最重要的一組 --------------------------------------------------


def test_value_white_keeps_sign_when_white_to_move() -> None:
    """白方走棋時，走棋方視角 == 白方視角，不該取負號。"""
    info = _info(chess.STARTING_FEN, "e2e4", 0.30)
    assert info.turn_before == chess.WHITE
    assert info.value_white() == pytest.approx(0.30)


def test_value_white_flips_sign_when_black_to_move() -> None:
    """黑方走棋時，走棋方視角要取負號才是白方視角。

    黑方覺得自己 +0.30（對黑方有利）→ 白方視角是 -0.30。
    """
    board = chess.Board()
    board.push_uci("e2e4")
    info = _info(board.fen(), "e7e5", 0.30)
    assert info.turn_before == chess.BLACK
    assert info.value_white() == pytest.approx(-0.30)


def test_eval_comment_sign_for_both_colours() -> None:
    """[%eval] 一律是白方視角：同樣的走棋方 value，黑方走時符號要相反。"""
    white_info = _info(chess.STARTING_FEN, "e2e4", 0.30)
    board = chess.Board()
    board.push_uci("e2e4")
    black_info = _info(board.fen(), "e7e5", 0.30)

    white_cp = value_to_cp(white_info.value_white())
    black_cp = value_to_cp(black_info.value_white())
    assert white_cp > 0, "白方覺得自己好，[%eval] 應為正"
    assert black_cp < 0, "黑方覺得自己好，[%eval] 應為負（白方視角）"
    assert white_cp == -black_cp

    # 檢查註解字串的正負號（別寫死小數位數：value 0.30 經 Leela 公式是 1.45 兵，
    # 不是 0.xx，寫死前綴只會測到自己的算術）
    white_comment = format_eval_comment(white_info)
    black_comment = format_eval_comment(black_info)
    assert white_comment.startswith("[%eval ") and "-" not in white_comment
    assert black_comment.startswith("[%eval -")


def test_winning_position_evaluates_positive_for_winner() -> None:
    """不論輪到誰走，白方大優的盤面 value_white 都該是正的。

    這條模擬的是實際對局裡最容易露出馬腳的情況：同一個局面，只差在誰走棋。
    """
    winning_for_white = "4k3/8/8/8/8/8/8/3QK3"
    # 白方走棋，白方覺得自己好 → +0.9
    white_to_move = _info(f"{winning_for_white} w - - 0 1", "d1d5", 0.9)
    # 黑方走棋，黑方覺得自己差 → -0.9（走棋方視角）
    black_to_move = _info(f"{winning_for_white} b - - 0 1", "e8f7", -0.9)

    assert white_to_move.value_white() > 0
    assert black_to_move.value_white() > 0
    assert white_to_move.value_white() == pytest.approx(black_to_move.value_white())


# --- cp 換算 ----------------------------------------------------------------


def test_value_to_cp_is_monotonic_and_signed() -> None:
    """cp 換算要單調遞增，且 0 對應 0。"""
    values = [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]
    cps = [value_to_cp(v) for v in values]
    assert cps == sorted(cps)
    assert value_to_cp(0.0) == 0
    assert value_to_cp(0.5) == -value_to_cp(-0.5)


def test_value_to_cp_clamps_extremes() -> None:
    """value = ±1 時 tan 會發散，必須夾住不能變成 inf 或爆炸。"""
    assert abs(value_to_cp(1.0)) <= 10000
    assert abs(value_to_cp(-1.0)) <= 10000
    assert abs(value_to_cp(5.0)) <= 10000


def test_cp_to_value_round_trip() -> None:
    """cp_to_value 是 value_to_cp 的反函數（第 5 節重訓 value 會用到）。"""
    for value in (-0.8, -0.3, 0.0, 0.3, 0.8):
        assert cp_to_value(value_to_cp(value)) == pytest.approx(value, abs=0.01)


def test_linear_cp_option() -> None:
    """線性換算是規格提供的另一個選項。"""
    assert value_to_cp(0.5, linear=True) == 300


# --- make_move_info ---------------------------------------------------------


def test_make_move_info_keeps_top_n_sorted() -> None:
    """policy_top 要依機率由高到低排序，且最多 POLICY_TOP_N 個。"""
    board = chess.Board()
    probs = {m: 1.0 / board.legal_moves.count() for m in board.legal_moves}
    best = chess.Move.from_uci("e2e4")
    probs[best] = 0.9
    info = make_move_info(board, best, 0.1, probs, elapsed_ms=12)

    assert len(info.policy_top) == POLICY_TOP_N
    assert info.policy_top[0][0] == "e2e4"
    values = [p for _, p in info.policy_top]
    assert values == sorted(values, reverse=True)
    assert info.san == "e4"
    assert info.fen_before == chess.STARTING_FEN
    assert info.visits is None       # Phase 1 沒有 MCTS


def test_make_move_info_does_not_mutate_board() -> None:
    """組 MoveInfo 不該把棋子走掉。"""
    board = chess.Board()
    before = board.fen()
    make_move_info(board, chess.Move.from_uci("e2e4"), 0.0, {}, 0)
    assert board.fen() == before


# --- PGN 輸出 ---------------------------------------------------------------


def test_build_game_has_required_headers() -> None:
    """必填標頭都要在，尤其 ModelCheckpoint（不然分不清是哪一版下的）。"""
    board = chess.Board()
    board.push_uci("e2e4")
    game = build_game(board, [_info(chess.STARTING_FEN, "e2e4", 0.1)], {})
    for key in REQUIRED_HEADERS:
        assert key in game.headers, f"缺少標頭 {key}"


def test_written_pgn_is_parseable_and_eval_survives(tmp_path: Path) -> None:
    """寫出去的 PGN 要能被讀回來，而且 [%eval] 要解析得出來。

    `python-chess` 的 `node.eval()` 用的就是 lichess 的 [%eval] 慣例，
    所以這條測試通過 = lichess 也認得。
    """
    board = chess.Board()
    infos = [_info(board.fen(), "e2e4", 0.30)]
    board.push_uci("e2e4")
    infos.append(_info(board.fen(), "e7e5", 0.20))
    board.push_uci("e7e5")

    path = tmp_path / "game.pgn"
    write_game(board, infos, {"White": "A", "Black": "B", "ModelCheckpoint": "x.pt"}, path)

    with open(path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    assert game is not None
    assert not game.errors

    nodes = list(game.mainline())
    assert len(nodes) == 2
    evals = [n.eval() for n in nodes]
    assert all(e is not None for e in evals)
    # 白方走 e4 時 value=+0.30 → 白方視角為正
    assert evals[0].white().score() > 0
    # 黑方走 e5 時 value=+0.20（黑方視角）→ 白方視角要變成負
    assert evals[1].white().score() < 0


def test_policy_comment_uses_san(tmp_path: Path) -> None:
    """policy 註解要用 SAN 顯示（存的是 UCI，寫的時候轉換）。"""
    info = MoveInfo(
        move=chess.Move.from_uci("e2e4"),
        san="e4",
        value=0.1,
        policy_top=[("e2e4", 0.41), ("g1f3", 0.22)],
        elapsed_ms=30,
        fen_before=chess.STARTING_FEN,
    )
    comment = format_policy_comment(info)
    assert "e4 .41" in comment
    assert "Nf3 .22" in comment      # g1f3 應該顯示成 SAN 的 Nf3
    assert "0.03s" in comment


def test_policy_comment_includes_visits_when_present() -> None:
    """MCTS 啟用時，註解要同時列出訪問次數（Phase 2 用）。"""
    info = MoveInfo(
        move=chess.Move.from_uci("e2e4"),
        san="e4",
        value=0.1,
        policy_top=[("e2e4", 0.41)],
        visits={"e2e4": 700, "d2d4": 100},
        elapsed_ms=1500,
        fen_before=chess.STARTING_FEN,
    )
    comment = format_policy_comment(info)
    assert "visits:" in comment and "e2e4:700" in comment
