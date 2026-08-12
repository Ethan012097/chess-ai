"""preprocess.py 的測試：用手寫的 3 局小 PGN 跑完整流程。

最重要的一條是 `result` 的正負號方向：白方贏的棋局，白方走棋的盤面 result 應為 +1，
黑方走棋的盤面應為 -1。這個符號寫反的話，value head 會學到完全相反的東西，
但 loss 看起來還是會下降，非常難察覺。
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest

from src.config import load_config
from src.encoding import decode_compact_to_planes, encode_board, encode_move
from src.preprocess import (
    POSITION_DTYPE,
    expand_inputs,
    parse_elo,
    parse_time_control_seconds,
    preprocess,
)

# Morphy vs. Duke of Brunswick / Count Isouard，巴黎歌劇院，1858（共 33 plies）
OPERA_GAME_MOVES = (
    "1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 "
    "6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 "
    "11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 "
    "15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8#"
)
EXPECTED_PLIES = 33


def _game_pgn(result: str, movetext: str, white_elo: int = 2400, black_elo: int = 2450) -> str:
    """組出一局 PGN 文字。"""
    return (
        f'[Event "Test Game"]\n'
        f'[Site "?"]\n'
        f'[Date "2024.01.01"]\n'
        f'[Round "1"]\n'
        f'[White "Alice"]\n'
        f'[Black "Bob"]\n'
        f'[Result "{result}"]\n'
        f'[WhiteElo "{white_elo}"]\n'
        f'[BlackElo "{black_elo}"]\n'
        f'[TimeControl "600+5"]\n'
        f'[Termination "Normal"]\n'
        f"\n"
        f"{movetext} {result}\n\n"
    )


@pytest.fixture
def three_game_pgn(tmp_path: Path) -> Path:
    """3 局測試 PGN：白勝 / 黑勝 / 和局，棋步相同只有 Result 不同。

    棋步一樣是刻意的：這樣同一個盤面在三局裡的 result 應該剛好是 +1 / -1 / 0，
    符號寫反會立刻被抓到。
    """
    pgn = tmp_path / "three_games.pgn"
    pgn.write_text(
        _game_pgn("1-0", OPERA_GAME_MOVES)
        + _game_pgn("0-1", OPERA_GAME_MOVES)
        + _game_pgn("1/2-1/2", OPERA_GAME_MOVES),
        encoding="utf-8",
    )
    return pgn


def _test_config(tmp_path: Path):
    """給測試用的設定：輸出到 tmp、val_ratio=0 讓盤面數完全可預期。"""
    out_dir = tmp_path / "processed"
    return load_config(
        overrides={
            "data": {
                "processed_dir": str(out_dir),
                "train_file": str(out_dir / "train.npy"),
                "val_file": str(out_dir / "val.npy"),
                "val_ratio": 0.0,
            }
        }
    )


def test_pgn_fixture_is_valid(three_game_pgn: Path) -> None:
    """先確認手寫的 PGN 本身沒打錯字，不然後面測到的都是假的。"""
    with open(three_game_pgn, encoding="utf-8") as f:
        results = []
        for _ in range(3):
            game = chess.pgn.read_game(f)
            assert game is not None
            assert not game.errors, f"PGN 解析有錯：{game.errors}"
            moves = list(game.mainline_moves())
            assert len(moves) == EXPECTED_PLIES, (
                f"棋步數應為 {EXPECTED_PLIES}，實際 {len(moves)}"
            )
            results.append(game.headers["Result"])
    assert results == ["1-0", "0-1", "1/2-1/2"]


def test_end_to_end_position_count(three_game_pgn: Path, tmp_path: Path) -> None:
    """驗證輸出筆數：跳過前 8 步與最後 2 步，每局應產生 33-8-2 = 23 個盤面。"""
    cfg = _test_config(tmp_path)
    n_train, n_val = preprocess([three_game_pgn], cfg, resume=False)

    expected_per_game = EXPECTED_PLIES - cfg.data.skip_opening_plies - cfg.data.skip_ending_plies
    assert expected_per_game == 23
    assert n_train == 3 * expected_per_game
    assert n_val == 0

    arr = np.load(cfg.resolve_path(cfg.data.train_file), mmap_mode="r")
    assert arr.dtype == POSITION_DTYPE
    assert arr.shape == (69,)


def test_result_sign_direction(three_game_pgn: Path, tmp_path: Path) -> None:
    """核心測試：result 必須是「當前走棋方」的視角。

    白方贏（1-0）的棋局裡，白方走棋的盤面 result = +1、黑方走棋的盤面 result = -1。
    黑方贏（0-1）剛好相反，和局全部是 0。
    """
    cfg = _test_config(tmp_path)
    preprocess([three_game_pgn], cfg, resume=False)
    arr = np.load(cfg.resolve_path(cfg.data.train_file), mmap_mode="r")

    per_game = 23
    start_ply = cfg.data.skip_opening_plies  # 8

    # 三局依序：白勝、黑勝、和局
    for game_idx, white_score in enumerate([1, -1, 0]):
        for i in range(per_game):
            ply = start_ply + i
            white_to_move = ply % 2 == 0
            expected = white_score if white_to_move else -white_score
            actual = int(arr[game_idx * per_game + i]["result"])
            assert actual == expected, (
                f"第 {game_idx + 1} 局、ply {ply}"
                f"（{'白方' if white_to_move else '黑方'}走棋）"
                f"的 result 應為 {expected}，實際 {actual}"
            )


def test_draw_positions_are_all_zero(three_game_pgn: Path, tmp_path: Path) -> None:
    """和局的 result 一律為 0。"""
    cfg = _test_config(tmp_path)
    preprocess([three_game_pgn], cfg, resume=False)
    arr = np.load(cfg.resolve_path(cfg.data.train_file), mmap_mode="r")
    draw_rows = arr[46:69]  # 第三局
    assert np.all(draw_rows["result"] == 0)


def test_stored_board_and_move_match_the_actual_game(
    three_game_pgn: Path, tmp_path: Path
) -> None:
    """存下來的盤面與著法，必須跟重播棋局得到的完全一致。

    這條測試把「前處理寫進硬碟的東西」跟「python-chess 眼中的真實盤面」綁在一起，
    是資料正確性的最後一道防線。
    """
    cfg = _test_config(tmp_path)
    preprocess([three_game_pgn], cfg, resume=False)
    arr = np.load(cfg.resolve_path(cfg.data.train_file), mmap_mode="r")

    # 重播第一局
    with open(three_game_pgn, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    assert game is not None
    moves = list(game.mainline_moves())

    board = game.board()
    start_ply = cfg.data.skip_opening_plies
    for ply, move in enumerate(moves):
        if start_ply <= ply < EXPECTED_PLIES - cfg.data.skip_ending_plies:
            row = arr[ply - start_ply]
            # 盤面：緊湊表示展開後要等於直接編碼
            restored = decode_compact_to_planes(
                row["pieces"], int(row["castling"]), int(row["ep_square"]), int(row["halfmove"])
            )
            np.testing.assert_array_equal(
                restored, encode_board(board), err_msg=f"ply {ply} 的盤面對不上"
            )
            # 著法
            assert int(row["move_index"]) == encode_move(move, board.turn), (
                f"ply {ply} 的 move_index 對不上"
            )
        board.push(move)


def test_position_dtype_is_70_bytes() -> None:
    """規格要求每個盤面 70 bytes，structured dtype 不可以被自動對齊撐大。"""
    assert POSITION_DTYPE.itemsize == 70


def test_low_elo_games_are_filtered(tmp_path: Path) -> None:
    """兩方 Elo 沒有都 ≥ 2000 的棋局要被擋掉。"""
    pgn = tmp_path / "low_elo.pgn"
    pgn.write_text(
        _game_pgn("1-0", OPERA_GAME_MOVES, white_elo=1500, black_elo=1500),
        encoding="utf-8",
    )
    cfg = _test_config(tmp_path)
    n_train, n_val = preprocess([pgn], cfg, resume=False)
    assert n_train == 0 and n_val == 0


def test_bullet_games_are_filtered(tmp_path: Path) -> None:
    """起始秒數 < 180 的 bullet 要被擋掉。"""
    pgn = tmp_path / "bullet.pgn"
    text = _game_pgn("1-0", OPERA_GAME_MOVES).replace(
        '[TimeControl "600+5"]', '[TimeControl "60+0"]'
    )
    pgn.write_text(text, encoding="utf-8")
    cfg = _test_config(tmp_path)
    n_train, _ = preprocess([pgn], cfg, resume=False)
    assert n_train == 0


def test_short_games_are_filtered(tmp_path: Path) -> None:
    """步數 < 20 的棋局要被擋掉。"""
    pgn = tmp_path / "short.pgn"
    pgn.write_text(_game_pgn("1-0", "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#"), encoding="utf-8")
    cfg = _test_config(tmp_path)
    n_train, _ = preprocess([pgn], cfg, resume=False)
    assert n_train == 0


def test_unfinished_games_are_filtered(tmp_path: Path) -> None:
    """Result 是 "*"（未完成）的棋局要被擋掉。"""
    pgn = tmp_path / "unfinished.pgn"
    pgn.write_text(_game_pgn("*", OPERA_GAME_MOVES), encoding="utf-8")
    cfg = _test_config(tmp_path)
    n_train, _ = preprocess([pgn], cfg, resume=False)
    assert n_train == 0


def test_max_positions_per_game_limit(three_game_pgn: Path, tmp_path: Path) -> None:
    """每盤棋最多取 max_positions_per_game 個盤面。"""
    cfg = _test_config(tmp_path)
    cfg.data.max_positions_per_game = 5
    n_train, _ = preprocess([three_game_pgn], cfg, resume=False)
    assert n_train == 3 * 5


def test_val_split_is_by_game(three_game_pgn: Path, tmp_path: Path) -> None:
    """train / val 依棋局切分：同一盤棋的盤面不可以被拆開。

    每局固定產生 23 個盤面，所以兩邊的盤面數都必須是 23 的倍數。
    """
    cfg = _test_config(tmp_path)
    cfg.data.val_ratio = 0.5
    n_train, n_val = preprocess([three_game_pgn], cfg, resume=False)
    assert n_train + n_val == 69
    assert n_train % 23 == 0, f"train 盤面數 {n_train} 不是整局的倍數，切分洩漏了"
    assert n_val % 23 == 0, f"val 盤面數 {n_val} 不是整局的倍數，切分洩漏了"


def test_max_positions_stops_early(three_game_pgn: Path, tmp_path: Path) -> None:
    """--max-positions 會在達標後停止。"""
    cfg = _test_config(tmp_path)
    n_train, _ = preprocess([three_game_pgn], cfg, max_positions=30, resume=False)
    assert 30 <= n_train <= 69


# --- 小工具函式 -------------------------------------------------------------


def test_parse_time_control_seconds() -> None:
    assert parse_time_control_seconds("600+5") == 600
    assert parse_time_control_seconds("180") == 180
    assert parse_time_control_seconds("-") is None
    assert parse_time_control_seconds(None) is None
    assert parse_time_control_seconds("亂寫") is None


def test_parse_elo() -> None:
    assert parse_elo("2400") == 2400
    assert parse_elo("?") is None
    assert parse_elo(None) is None


def test_expand_inputs_handles_globs(tmp_path: Path) -> None:
    """Windows 的 PowerShell 不會展開 *，要靠 expand_inputs 自己做。"""
    (tmp_path / "a.pgn").write_text("", encoding="utf-8")
    (tmp_path / "b.pgn").write_text("", encoding="utf-8")
    found = expand_inputs([str(tmp_path / "*.pgn")])
    assert len(found) == 2
    # 重複給同一個檔案只算一次
    p = str(tmp_path / "a.pgn")
    assert len(expand_inputs([p, p])) == 1
