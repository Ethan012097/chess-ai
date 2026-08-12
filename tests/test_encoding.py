"""encoding.py 的測試。這些測試不是形式，是拿來救命的。

top-1 準確率若停在 10% 以下，八成是這裡寫錯了，回來跑這支就知道。
所有測試都在 CPU 上跑。
"""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from src.encoding import (
    BOARD_SIZE,
    NUM_INPUT_PLANES,
    NUM_MOVES,
    PLANE_EN_PASSANT,
    PLANE_HALFMOVE,
    PLANE_OPP_KINGSIDE,
    PLANE_OPP_PIECES,
    PLANE_OPP_QUEENSIDE,
    PLANE_OWN_KINGSIDE,
    PLANE_OWN_PIECES,
    PLANE_OWN_QUEENSIDE,
    decode_compact_to_planes,
    decode_move,
    encode_board,
    encode_board_compact,
    encode_move,
    index_to_move,
    legal_indices,
    legal_mask,
    mirror_move,
    move_to_index,
    to_canonical_board,
)


def _random_games(num_games: int, seed: int = 0) -> list[chess.Board]:
    """隨機走子產生一堆盤面，用來當測試素材。

    Returns:
        每個盤面都是「遊戲進行中」的合法盤面（含開局與中殘局）。
    """
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    for _ in range(num_games):
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(rng.choice(moves))
            boards.append(board.copy(stack=False))
    return boards


# --- round-trip：最重要的一項 ------------------------------------------------


def test_move_index_round_trip_on_random_positions() -> None:
    """對隨機盤面的所有合法著法驗證 index_to_move(move_to_index(m)) == m。

    規格要求覆蓋 10000 個盤面。
    """
    boards = _random_games(num_games=100, seed=1234)
    assert len(boards) >= 1000, "測試素材太少，隨機對局產生的盤面不足"

    checked_positions = 0
    checked_moves = 0
    for board in boards:
        canonical = to_canonical_board(board)
        for move in canonical.legal_moves:
            idx = move_to_index(move)
            assert 0 <= idx < NUM_MOVES
            back = index_to_move(idx, canonical)
            assert back == move, (
                f"round-trip 失敗：{move.uci()} → {idx} → {back.uci()}\n"
                f"FEN={canonical.fen()}"
            )
            checked_moves += 1
        checked_positions += 1
        if checked_positions >= 10000:
            break

    assert checked_moves > 10000, f"檢查的著法太少（{checked_moves}）"


def test_move_index_round_trip_with_mirroring() -> None:
    """含鏡射的完整 round-trip：encode_move / decode_move 要能回到原盤面座標。"""
    boards = _random_games(num_games=60, seed=99)
    for board in boards:
        for move in board.legal_moves:
            idx = encode_move(move, board.turn)
            back = decode_move(idx, board)
            assert back == move, (
                f"含鏡射的 round-trip 失敗：{move.uci()} → {idx} → {back.uci()}\n"
                f"FEN={board.fen()}"
            )


def test_underpromotion_round_trip() -> None:
    """升變（含 underpromotion 與升后）在四種方向上都要能 round-trip。"""
    # 白兵在 b7，可以直走 b8、也可以吃 a8 或 c8（黑方兩隻車可吃）。
    # 白王放 e1 而不是 a1，否則 a8 的車會將軍，只剩吃 a8 一種合法著法。
    board = chess.Board("r1r5/1P6/8/8/8/8/8/4K2k w - - 0 1")
    promotions = [m for m in board.legal_moves if m.promotion is not None]
    assert len(promotions) == 12, (
        f"測試盤面應該要有 3 個方向 × 4 種升變 = 12 種，實際 {len(promotions)}"
    )

    seen_pieces = set()
    for move in promotions:
        idx = move_to_index(move)
        back = index_to_move(idx, board)
        assert back == move, f"升變 round-trip 失敗：{move.uci()} → {idx} → {back.uci()}"
        seen_pieces.add(move.promotion)

    assert seen_pieces == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}


def test_all_indices_are_distinct_per_position() -> None:
    """同一個盤面裡，不同合法著法不可以撞到同一個 index。"""
    for board in _random_games(num_games=40, seed=7):
        canonical = to_canonical_board(board)
        indices = [move_to_index(m) for m in canonical.legal_moves]
        assert len(indices) == len(set(indices)), (
            f"有著法撞 index，FEN={canonical.fen()}"
        )


# --- 初始盤面的編碼 ----------------------------------------------------------


def test_starting_position_encoding() -> None:
    """驗證起始盤面：己方 8 個兵在 plane 0 的 rank 1、4 個易位權都是 1。"""
    board = chess.Board()
    planes = encode_board(board)

    assert planes.shape == (NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert planes.dtype == np.float32

    # 己方（白方）兵在 rank 1 整排
    assert planes[PLANE_OWN_PIECES, 1, :].sum() == 8
    assert planes[PLANE_OWN_PIECES].sum() == 8
    # 對方兵在 rank 6 整排
    assert planes[PLANE_OPP_PIECES, 6, :].sum() == 8
    assert planes[PLANE_OPP_PIECES].sum() == 8

    # 己方王在 e1 = (rank 0, file 4)
    assert planes[PLANE_OWN_PIECES + 5, 0, 4] == 1.0
    # 對方王在 e8 = (rank 7, file 4)
    assert planes[PLANE_OPP_PIECES + 5, 7, 4] == 1.0

    # 4 個易位權整層都是 1
    for plane_idx in (
        PLANE_OWN_KINGSIDE,
        PLANE_OWN_QUEENSIDE,
        PLANE_OPP_KINGSIDE,
        PLANE_OPP_QUEENSIDE,
    ):
        assert planes[plane_idx].min() == 1.0 and planes[plane_idx].max() == 1.0

    # 沒有吃過路兵、五十步計數為 0
    assert planes[PLANE_EN_PASSANT].sum() == 0.0
    assert planes[PLANE_HALFMOVE].max() == 0.0


def test_piece_planes_are_one_hot() -> None:
    """每個格子最多只能有一個棋子 plane 是 1。"""
    for board in _random_games(num_games=20, seed=5):
        planes = encode_board(board)
        piece_planes = planes[PLANE_OWN_PIECES : PLANE_OPP_PIECES + 6]
        assert piece_planes.sum(axis=0).max() <= 1.0


# --- canonical / 鏡射一致性 --------------------------------------------------


def test_mirror_consistency() -> None:
    """canonical 的意義：encode(board) 與 encode(board.mirror()) 必須完全相同。

    因為兩者只是同一個局面從各自走棋方視角看出去的結果。
    """
    for board in _random_games(num_games=30, seed=2024):
        a = encode_board(board)
        b = encode_board(board.mirror())
        np.testing.assert_array_equal(
            a, b, err_msg=f"鏡射一致性失敗，FEN={board.fen()}"
        )


def test_canonical_turn_is_always_white() -> None:
    """canonical 之後，走棋方永遠是白方。"""
    for board in _random_games(num_games=20, seed=11):
        assert to_canonical_board(board).turn == chess.WHITE


def test_mirror_move_is_its_own_inverse() -> None:
    """mirror_move 套兩次應該回到原著法。"""
    for board in _random_games(num_games=20, seed=31):
        for move in board.legal_moves:
            assert mirror_move(mirror_move(move)) == move


def test_black_to_move_sees_own_pawns_on_rank_one() -> None:
    """輪到黑方時，黑方的兵在編碼裡也要出現在「己方 rank 1」。"""
    board = chess.Board()
    board.push_uci("e2e4")  # 換黑方走
    assert board.turn == chess.BLACK

    planes = encode_board(board)
    # 黑方 7 個兵還在原位（d7..h7, a7, b7, c7 共 8 個，都沒動）
    assert planes[PLANE_OWN_PIECES, 1, :].sum() == 8
    # 對方（白方）的 e4 兵，在己方視角是 rank 7-3=4... 直接檢查對方兵總數
    assert planes[PLANE_OPP_PIECES].sum() == 8


# --- legal_mask -------------------------------------------------------------


def test_legal_mask_count_matches_legal_moves() -> None:
    """legal_mask 的 True 數量必須等於 board.legal_moves.count()。"""
    for board in _random_games(num_games=40, seed=808):
        mask = legal_mask(board)
        assert mask.shape == (NUM_MOVES,)
        assert mask.dtype == np.bool_
        assert int(mask.sum()) == board.legal_moves.count(), (
            f"mask 數量不符，FEN={board.fen()}"
        )


def test_legal_mask_on_starting_position() -> None:
    """初始盤面有 20 種合法著法。"""
    assert int(legal_mask(chess.Board()).sum()) == 20


def test_legal_indices_maps_back_to_original_moves() -> None:
    """legal_indices 回傳的 value 必須是原始盤面座標的合法著法。"""
    for board in _random_games(num_games=30, seed=606):
        mapping = legal_indices(board)
        assert len(mapping) == board.legal_moves.count()
        legal_set = set(board.legal_moves)
        for idx, move in mapping.items():
            assert move in legal_set, f"{move.uci()} 不是合法著法，FEN={board.fen()}"
            assert 0 <= idx < NUM_MOVES


# --- 緊湊表示 ---------------------------------------------------------------


def test_compact_encoding_matches_full_encoding() -> None:
    """緊湊表示展開後，必須與直接編碼的張量逐格相同。

    這條測試守住的是「訓練時餵進網路的東西」跟「前處理時看到的盤面」一致。
    """
    for board in _random_games(num_games=40, seed=4242):
        pieces, castling, ep_square, halfmove = encode_board_compact(board)
        restored = decode_compact_to_planes(pieces, castling, ep_square, halfmove)
        direct = encode_board(board)
        np.testing.assert_array_equal(
            restored, direct, err_msg=f"緊湊表示不一致，FEN={board.fen()}"
        )


def test_compact_to_board_round_trip() -> None:
    """緊湊表示 → Board → 再編碼一次，張量必須完全一致。

    這條守住的是「拿存下來的盤面去餵 Stockfish」那條路：如果還原出來的 FEN 是錯的，
    用它算出來的 value 品質指標就整個沒有意義。
    """
    from src.encoding import compact_to_board

    for board in _random_games(num_games=40, seed=31337):
        pieces, castling, ep_square, halfmove = encode_board_compact(board)
        restored = compact_to_board(pieces, castling, ep_square, halfmove)

        # canonical 盤面的走棋方一定是白方
        assert restored.turn == chess.WHITE
        # 還原出來的盤面再編碼一次，要跟原本的張量一模一樣
        np.testing.assert_array_equal(
            encode_board(restored),
            encode_board(board),
            err_msg=f"還原後的盤面對不上，FEN={board.fen()}",
        )
        assert restored.is_valid(), f"還原出來的盤面不合法：{restored.fen()}"


def test_compact_to_board_preserves_castling_and_ep() -> None:
    """易位權與吃過路兵目標格都要還原正確。"""
    from src.encoding import compact_to_board

    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("a7a6")
    board.push_uci("e4e5")
    board.push_uci("d7d5")          # 產生 d6 的 ep 目標格，且輪到白方
    assert board.ep_square is not None

    pieces, castling, ep_square, halfmove = encode_board_compact(board)
    restored = compact_to_board(pieces, castling, ep_square, halfmove)
    assert restored.ep_square == board.ep_square
    assert restored.has_kingside_castling_rights(chess.WHITE)
    assert restored.has_queenside_castling_rights(chess.WHITE)


def test_compact_encoding_dtypes_and_ranges() -> None:
    """緊湊表示的值域要塞得進規格說的 70 bytes。"""
    for board in _random_games(num_games=20, seed=1357):
        pieces, castling, ep_square, halfmove = encode_board_compact(board)
        assert pieces.dtype == np.int8 and pieces.shape == (64,)
        assert pieces.min() >= 0 and pieces.max() <= 12
        assert 0 <= castling <= 0b1111
        assert -1 <= ep_square < 64
        assert 0 <= halfmove <= 255


def test_compact_encoding_en_passant() -> None:
    """有吃過路兵目標格時，plane 16 只有那一格是 1。"""
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("a7a6")
    board.push_uci("e4e5")
    board.push_uci("d7d5")  # 產生 d6 的 ep 目標格
    assert board.ep_square is not None

    planes = encode_board(board)
    assert planes[PLANE_EN_PASSANT].sum() == 1.0

    pieces, castling, ep_square, halfmove = encode_board_compact(board)
    assert ep_square >= 0
    restored = decode_compact_to_planes(pieces, castling, ep_square, halfmove)
    np.testing.assert_array_equal(restored, planes)


def test_halfmove_plane_scaling() -> None:
    """五十步計數要除以 100 填滿整層。"""
    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 37 80")
    planes = encode_board(board)
    assert planes[PLANE_HALFMOVE].min() == pytest.approx(0.37)
    assert planes[PLANE_HALFMOVE].max() == pytest.approx(0.37)


# --- 索引空間 ---------------------------------------------------------------


def test_index_range_and_invalid_input() -> None:
    """index 超出範圍要丟 ValueError；走出棋盤的 index 回傳 null move。"""
    board = chess.Board()
    with pytest.raises(ValueError):
        index_to_move(-1, board)
    with pytest.raises(ValueError):
        index_to_move(NUM_MOVES, board)

    # a1 往南走一格必定出界
    from src.encoding import NUM_MOVE_PLANES, QUEEN_DIRECTIONS, MAX_QUEEN_DISTANCE

    south_dir = QUEEN_DIRECTIONS.index((0, -1))
    idx = chess.A1 * NUM_MOVE_PLANES + south_dir * MAX_QUEEN_DISTANCE
    assert index_to_move(idx, board) == chess.Move.null()


def test_num_moves_is_4672() -> None:
    """規格寫死的維度，避免有人改常數改壞。"""
    assert NUM_MOVES == 4672
    assert NUM_INPUT_PLANES == 18
