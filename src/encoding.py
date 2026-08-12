"""盤面編碼與著法編碼（Phase 1 / Phase 2 共用）。

這是整個專案最容易寫錯的地方，改動前請先跑 `pytest tests/test_encoding.py`。

兩個核心概念：

1. **Canonical orientation（正規化視角）**
   永遠從「輪到走棋的一方」的視角編碼。輪到黑方時先把盤面上下鏡射並交換顏色
   （python-chess 的 `board.mirror()`），使「自己」永遠是白方、永遠往 rank 增加的
   方向前進。這樣網路只要學一種視角，資料效率加倍，Phase 2 的 MCTS 也能直接沿用。

2. **AlphaZero 的 4672 維著法編碼**
   4672 = 64 個起始格 × 73 種移動類型。移動類型分三段：
     - 0–55  queen moves：8 個方向 × 1–7 格距離
     - 56–63 knight moves：8 個方向
     - 64–72 underpromotion：3 個方向（直走 / 左吃 / 右吃）× 3 種棋子（N, B, R）
   升變成后不另外編碼，走 queen moves 那一格即可。
"""

from __future__ import annotations

import argparse

import chess
import numpy as np

# --- 具名常數（不要在程式碼裡出現裸的數字） ---------------------------------

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE          # 64

NUM_PIECE_TYPES = 6                            # P, N, B, R, Q, K
NUM_INPUT_PLANES = 18                          # 見下方 §輸入張量
PLANE_OWN_PIECES = 0                           # planes 0–5：己方棋子
PLANE_OPP_PIECES = 6                           # planes 6–11：對方棋子
PLANE_OWN_KINGSIDE = 12                        # 己方王翼易位權
PLANE_OWN_QUEENSIDE = 13                       # 己方后翼易位權
PLANE_OPP_KINGSIDE = 14                        # 對方王翼易位權
PLANE_OPP_QUEENSIDE = 15                       # 對方后翼易位權
PLANE_EN_PASSANT = 16                          # 吃過路兵目標格
PLANE_HALFMOVE = 17                            # 五十步計數 / 100.0

HALFMOVE_SCALE = 100.0                         # 五十步規則以 100 為分母正規化

NUM_QUEEN_DIRECTIONS = 8
MAX_QUEEN_DISTANCE = 7
NUM_QUEEN_MOVES = NUM_QUEEN_DIRECTIONS * MAX_QUEEN_DISTANCE   # 56
NUM_KNIGHT_MOVES = 8                                          # 56–63
NUM_UNDERPROMO_DIRECTIONS = 3                                 # 直走 / 左吃 / 右吃
NUM_UNDERPROMO_PIECES = 3                                     # N, B, R
NUM_UNDERPROMO_MOVES = NUM_UNDERPROMO_DIRECTIONS * NUM_UNDERPROMO_PIECES  # 9

NUM_MOVE_PLANES = NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_UNDERPROMO_MOVES  # 73
NUM_MOVES = NUM_SQUARES * NUM_MOVE_PLANES                                    # 4672

QUEEN_PLANE_START = 0
KNIGHT_PLANE_START = NUM_QUEEN_MOVES                       # 56
UNDERPROMO_PLANE_START = KNIGHT_PLANE_START + NUM_KNIGHT_MOVES  # 64

# queen moves 的 8 個方向，順序固定為 (df, dr)：
# N, NE, E, SE, S, SW, W, NW（df = file 位移, dr = rank 位移）
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
)

# knight moves 的 8 個方向 (df, dr)
KNIGHT_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# underpromotion 的 3 個 file 位移：-1 = 往左吃, 0 = 直走, +1 = 往右吃
UNDERPROMO_FILE_DELTAS: tuple[int, ...] = (-1, 0, 1)
# underpromotion 的 3 種棋子（升后走 queen moves，不在這裡）
UNDERPROMO_PIECES: tuple[chess.PieceType, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

# python-chess 的 PieceType 是 1..6 (PAWN..KING)，減 1 就是 plane 偏移
PIECE_TYPE_TO_PLANE_OFFSET: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


# --- 查表：把 (方向, 距離) 之類的組合預先算好，round-trip 才不會靠 if/else 疊出錯 ---


def _build_queen_lookup() -> dict[tuple[int, int], int]:
    """(df, dr) → queen move plane index（0–55）。只收 8 方向的直線位移。"""
    table: dict[tuple[int, int], int] = {}
    for dir_idx, (dfile, drank) in enumerate(QUEEN_DIRECTIONS):
        for distance in range(1, MAX_QUEEN_DISTANCE + 1):
            table[(dfile * distance, drank * distance)] = (
                dir_idx * MAX_QUEEN_DISTANCE + (distance - 1)
            )
    return table


def _build_knight_lookup() -> dict[tuple[int, int], int]:
    """(df, dr) → knight move plane index（56–63）。"""
    return {
        delta: KNIGHT_PLANE_START + i for i, delta in enumerate(KNIGHT_DIRECTIONS)
    }


_QUEEN_LOOKUP = _build_queen_lookup()
_KNIGHT_LOOKUP = _build_knight_lookup()

# 反查表：plane index → (df, dr)，index_to_move 用
_QUEEN_PLANE_TO_DELTA: dict[int, tuple[int, int]] = {
    v: k for k, v in _QUEEN_LOOKUP.items()
}
_KNIGHT_PLANE_TO_DELTA: dict[int, tuple[int, int]] = {
    v: k for k, v in _KNIGHT_LOOKUP.items()
}


# --- 盤面編碼 ---------------------------------------------------------------


def to_canonical_board(board: chess.Board) -> chess.Board:
    """把盤面轉成 canonical orientation（走棋方永遠是白方、永遠往上走）。

    Args:
        board: 任意盤面。

    Returns:
        新的 Board。輪到白方時回傳 `board.copy()`；輪到黑方時回傳 `board.mirror()`
        （上下鏡射 + 交換顏色，走棋方變成白方）。
    """
    return board.copy(stack=False) if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> np.ndarray:
    """把盤面編碼成 (18, 8, 8) float32 張量（已套用 canonical orientation）。

    索引約定為 `plane[rank][file]`，`rank=0` 是己方底線。

    Args:
        board: 任意盤面（函式內部自己處理鏡射，呼叫端不用先轉）。

    Returns:
        shape (18, 8, 8) 的 float32 陣列，值域 0/1（plane 17 是 0~1 的連續值）。
    """
    canonical = to_canonical_board(board)
    planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # planes 0–11：棋子位置。canonical 之後「己方」一定是白方。
    for square, piece in canonical.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        base = PLANE_OWN_PIECES if piece.color == chess.WHITE else PLANE_OPP_PIECES
        planes[base + PIECE_TYPE_TO_PLANE_OFFSET[piece.piece_type], rank, file] = 1.0

    # planes 12–15：易位權，整層填 0 或 1
    if canonical.has_kingside_castling_rights(chess.WHITE):
        planes[PLANE_OWN_KINGSIDE, :, :] = 1.0
    if canonical.has_queenside_castling_rights(chess.WHITE):
        planes[PLANE_OWN_QUEENSIDE, :, :] = 1.0
    if canonical.has_kingside_castling_rights(chess.BLACK):
        planes[PLANE_OPP_KINGSIDE, :, :] = 1.0
    if canonical.has_queenside_castling_rights(chess.BLACK):
        planes[PLANE_OPP_QUEENSIDE, :, :] = 1.0

    # plane 16：吃過路兵目標格（只有該格為 1）
    if canonical.ep_square is not None:
        planes[
            PLANE_EN_PASSANT,
            chess.square_rank(canonical.ep_square),
            chess.square_file(canonical.ep_square),
        ] = 1.0

    # plane 17：五十步計數 / 100，整層填同一個值
    planes[PLANE_HALFMOVE, :, :] = canonical.halfmove_clock / HALFMOVE_SCALE

    return planes


def encode_board_compact(board: chess.Board) -> tuple[np.ndarray, int, int, int]:
    """把盤面編碼成前處理要存進硬碟的緊湊表示（已鏡射）。

    存 (18,8,8) float32 是 4.6 KB/盤面，1500 萬盤面會爆硬碟；所以只存這個緊湊
    版本（70 bytes/盤面），到 `Dataset.__getitem__` 才展開成張量。

    Args:
        board: 任意盤面。

    Returns:
        (pieces, castling, ep_square, halfmove)
          pieces:   int8[64]，0=空、1..6=己方 PNBRQK、7..12=對方 PNBRQK（已鏡射）
                    索引為 `square = rank * 8 + file`
          castling: uint8，4 個 bit（bit0 己方王翼、bit1 己方后翼、
                    bit2 對方王翼、bit3 對方后翼）
          ep_square: int8，-1 代表沒有
          halfmove: uint8，超過 255 就截斷（五十步規則本來就上限 100）
    """
    canonical = to_canonical_board(board)

    pieces = np.zeros(NUM_SQUARES, dtype=np.int8)
    for square, piece in canonical.piece_map().items():
        offset = PIECE_TYPE_TO_PLANE_OFFSET[piece.piece_type]  # 0..5
        code = offset + 1 if piece.color == chess.WHITE else offset + 1 + NUM_PIECE_TYPES
        pieces[square] = code

    castling = 0
    if canonical.has_kingside_castling_rights(chess.WHITE):
        castling |= 1 << 0
    if canonical.has_queenside_castling_rights(chess.WHITE):
        castling |= 1 << 1
    if canonical.has_kingside_castling_rights(chess.BLACK):
        castling |= 1 << 2
    if canonical.has_queenside_castling_rights(chess.BLACK):
        castling |= 1 << 3

    ep_square = -1 if canonical.ep_square is None else int(canonical.ep_square)
    halfmove = min(int(canonical.halfmove_clock), 255)

    return pieces, castling, ep_square, halfmove


def compact_to_board(
    pieces: np.ndarray,
    castling: int,
    ep_square: int,
    halfmove: int,
) -> chess.Board:
    """把緊湊表示還原成 `chess.Board`（canonical 視角，走棋方永遠是白方）。

    這是 `encode_board_compact` 的反函數。用途是拿前處理存下來的盤面去餵別的
    工具（例如用 Stockfish 重新評估 val 集），因為那些工具要的是 FEN 不是張量。

    **注意還原出來的是 canonical 盤面**：原本輪到黑方的局面會以「顏色互換 + 上下
    鏡射」的形式出現。這對評估沒有影響（局面的優劣不會因為鏡射而改變），
    而且走棋方一律是白方，剛好對應 value 的「當前走棋方視角」。

    因為緊湊表示沒有存回合數，`fullmove_number` 一律填 1。

    Args:
        pieces: int8[64]，見 `encode_board_compact`。
        castling: uint8，4 個 bit。
        ep_square: int8，-1 代表沒有。
        halfmove: uint8。

    Returns:
        chess.Board，`turn` 一定是 WHITE。
    """
    # code 1..6 → 白方 PNBRQK，7..12 → 黑方 pnbrqk
    symbols = "PNBRQKpnbrqk"
    codes = np.asarray(pieces, dtype=np.int64).reshape(NUM_SQUARES)

    rows: list[str] = []
    for rank in range(BOARD_SIZE - 1, -1, -1):     # FEN 從第 8 排寫到第 1 排
        row = ""
        empty = 0
        for file in range(BOARD_SIZE):
            code = int(codes[rank * BOARD_SIZE + file])
            if code == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += symbols[code - 1]
        if empty:
            row += str(empty)
        rows.append(row)

    rights = ""
    if castling & (1 << 0):
        rights += "K"
    if castling & (1 << 1):
        rights += "Q"
    if castling & (1 << 2):
        rights += "k"
    if castling & (1 << 3):
        rights += "q"

    ep = chess.square_name(int(ep_square)) if ep_square >= 0 else "-"
    fen = f"{'/'.join(rows)} w {rights or '-'} {ep} {int(halfmove)} 1"
    return chess.Board(fen)


def decode_compact_to_planes(
    pieces: np.ndarray,
    castling: int,
    ep_square: int,
    halfmove: int,
) -> np.ndarray:
    """把緊湊表示展開回 (18, 8, 8) float32 張量。

    這支會在 `Dataset.__getitem__` 裡被每筆資料呼叫一次，所以**全部用 NumPy
    向量化**，不能碰 python-chess，否則 GPU 會被 DataLoader 餓死。

    Args:
        pieces: int8[64]，見 `encode_board_compact`。
        castling: uint8，4 個 bit。
        ep_square: int8，-1 代表沒有。
        halfmove: uint8。

    Returns:
        shape (18, 8, 8) 的 float32 陣列，與 `encode_board` 的輸出完全一致。
    """
    planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # planes 0–11：用 one-hot 散射。code 1..12 對應 plane 0..11。
    codes = np.asarray(pieces, dtype=np.int64).reshape(NUM_SQUARES)
    occupied = codes > 0
    if occupied.any():
        squares = np.nonzero(occupied)[0]
        plane_idx = codes[squares] - 1          # 1..12 → 0..11
        rank_idx = squares // BOARD_SIZE
        file_idx = squares % BOARD_SIZE
        planes[plane_idx, rank_idx, file_idx] = 1.0

    # planes 12–15：易位權
    if castling & (1 << 0):
        planes[PLANE_OWN_KINGSIDE, :, :] = 1.0
    if castling & (1 << 1):
        planes[PLANE_OWN_QUEENSIDE, :, :] = 1.0
    if castling & (1 << 2):
        planes[PLANE_OPP_KINGSIDE, :, :] = 1.0
    if castling & (1 << 3):
        planes[PLANE_OPP_QUEENSIDE, :, :] = 1.0

    # plane 16：吃過路兵
    if ep_square >= 0:
        planes[PLANE_EN_PASSANT, ep_square // BOARD_SIZE, ep_square % BOARD_SIZE] = 1.0

    # plane 17：五十步計數
    planes[PLANE_HALFMOVE, :, :] = float(halfmove) / HALFMOVE_SCALE

    return planes


# --- 著法編碼 ---------------------------------------------------------------


def _is_underpromotion(move: chess.Move) -> bool:
    """升變成 N / B / R 才算 underpromotion；升后走 queen moves。"""
    return move.promotion is not None and move.promotion != chess.QUEEN


def move_to_index(move: chess.Move) -> int:
    """把著法轉成 0..4671 的 index。

    **輸入必須是 canonical 視角的著法**（也就是走棋方為白方的盤面上的著法）。
    如果原盤面輪到黑方，請先用 `mirror_move` 轉換。

    Args:
        move: canonical 視角下的合法著法。

    Returns:
        0 <= index < 4672。

    Raises:
        ValueError: 著法不符合任何一種移動類型（通常代表傳進了非法著法）。
    """
    from_sq = move.from_square
    to_sq = move.to_square
    dfile = chess.square_file(to_sq) - chess.square_file(from_sq)
    drank = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if _is_underpromotion(move):
        # underpromotion：只可能是兵往前一格（含斜吃），所以只看 file 位移
        try:
            dir_idx = UNDERPROMO_FILE_DELTAS.index(dfile)
            piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        except ValueError as exc:
            raise ValueError(f"無法編碼的 underpromotion：{move.uci()}") from exc
        plane = (
            UNDERPROMO_PLANE_START
            + dir_idx * NUM_UNDERPROMO_PIECES
            + piece_idx
        )
    elif (dfile, drank) in _KNIGHT_LOOKUP:
        plane = _KNIGHT_LOOKUP[(dfile, drank)]
    elif (dfile, drank) in _QUEEN_LOOKUP:
        plane = _QUEEN_LOOKUP[(dfile, drank)]
    else:
        raise ValueError(
            f"無法編碼的著法：{move.uci()}（位移 df={dfile}, dr={drank}）"
        )

    return from_sq * NUM_MOVE_PLANES + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """把 index 轉回著法（需要盤面才能判斷是不是升后）。

    **`board` 必須是 canonical 視角的盤面**（走棋方為白方）。

    為什麼需要 board：queen moves 的 plane 沒有記錄升變資訊。兵走到第 8 排時，
    同一個 index 要解讀成「升后」；其他情況則是普通移動。用盤面查出起始格是不是
    兵，就能決定要不要補上 `promotion=QUEEN`。

    Args:
        index: 0..4671。
        board: canonical 視角的盤面。

    Returns:
        chess.Move。若 index 對應的目標格超出棋盤，回傳 `chess.Move.null()`。

    Raises:
        ValueError: index 超出範圍。
    """
    if not 0 <= index < NUM_MOVES:
        raise ValueError(f"index 必須在 0..{NUM_MOVES - 1}，收到 {index}")

    from_sq = index // NUM_MOVE_PLANES
    plane = index % NUM_MOVE_PLANES
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)

    promotion: chess.PieceType | None = None

    if plane >= UNDERPROMO_PLANE_START:
        offset = plane - UNDERPROMO_PLANE_START
        dir_idx, piece_idx = divmod(offset, NUM_UNDERPROMO_PIECES)
        dfile = UNDERPROMO_FILE_DELTAS[dir_idx]
        drank = 1  # canonical 視角下，兵永遠往 rank 增加的方向升變
        promotion = UNDERPROMO_PIECES[piece_idx]
    elif plane >= KNIGHT_PLANE_START:
        dfile, drank = _KNIGHT_PLANE_TO_DELTA[plane]
    else:
        dfile, drank = _QUEEN_PLANE_TO_DELTA[plane]

    to_file = from_file + dfile
    to_rank = from_rank + drank
    if not (0 <= to_file < BOARD_SIZE and 0 <= to_rank < BOARD_SIZE):
        return chess.Move.null()  # 走出棋盤，這個 index 在此盤面無意義

    to_sq = chess.square(to_file, to_rank)

    # queen moves 打到底線的兵 → 補上升后
    if promotion is None and to_rank == BOARD_SIZE - 1:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def mirror_move(move: chess.Move) -> chess.Move:
    """把著法在上下鏡射後的盤面之間轉換（自己是自己的反函式）。

    `board.mirror()` 會把 square 上下翻，python-chess 提供 `chess.square_mirror`。
    升變棋子種類不變。

    Args:
        move: 任一著法。

    Returns:
        鏡射後的著法。
    """
    if move == chess.Move.null():
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """把「原始盤面」上的著法轉成 canonical index，自動處理鏡射。

    Args:
        move: 原始盤面上的合法著法。
        turn: 原始盤面輪到誰走（`board.turn`）。輪到黑方時會先鏡射。

    Returns:
        0..4671 的 index。
    """
    canonical_move = move if turn == chess.WHITE else mirror_move(move)
    return move_to_index(canonical_move)


def decode_move(index: int, board: chess.Board) -> chess.Move:
    """把 canonical index 轉回「原始盤面」上的著法，自動處理鏡射還原。

    Args:
        index: 0..4671。
        board: 原始盤面（不是 canonical 盤面）。

    Returns:
        原始盤面座標系下的著法；若無法對應則回傳 `chess.Move.null()`。
    """
    canonical_board = to_canonical_board(board)
    canonical_move = index_to_move(index, canonical_board)
    if canonical_move == chess.Move.null():
        return canonical_move
    return canonical_move if board.turn == chess.WHITE else mirror_move(canonical_move)


def legal_mask(board: chess.Board) -> np.ndarray:
    """算出 canonical 視角下的合法著法遮罩。

    推論時用來把非法著法的 logit 設成 -inf。

    Args:
        board: 原始盤面（函式內部自己鏡射）。

    Returns:
        shape (4672,) 的 bool 陣列，True 代表該 index 是合法著法。
        True 的數量等於 `board.legal_moves.count()`。
    """
    mask = np.zeros(NUM_MOVES, dtype=bool)
    canonical = to_canonical_board(board)
    for move in canonical.legal_moves:
        mask[move_to_index(move)] = True
    return mask


def legal_indices(board: chess.Board) -> dict[int, chess.Move]:
    """回傳 {canonical index: 原始盤面的著法}，下棋時用來一次做完 mask + 還原。

    比「先 legal_mask 再 decode_move」快，也不會有 decode 對不回去的風險。

    Args:
        board: 原始盤面。

    Returns:
        dict，key 是 0..4671 的 index，value 是**原始盤面座標**的著法。
    """
    canonical = to_canonical_board(board)
    is_white = board.turn == chess.WHITE
    result: dict[int, chess.Move] = {}
    for move in canonical.legal_moves:
        result[move_to_index(move)] = move if is_white else mirror_move(move)
    return result


def main() -> None:
    """`python -m src.encoding` 拿一個盤面做示範，順便當作快速自檢。"""
    parser = argparse.ArgumentParser(description="盤面 / 著法編碼的示範與自檢")
    parser.add_argument(
        "--fen",
        type=str,
        default=chess.STARTING_FEN,
        help="要編碼的盤面 FEN（預設為初始盤面）",
    )
    args = parser.parse_args()

    board = chess.Board(args.fen)
    planes = encode_board(board)
    mask = legal_mask(board)

    print(f"FEN            : {board.fen()}")
    print(f"輪到           : {'白方' if board.turn == chess.WHITE else '黑方'}")
    print(f"輸入張量 shape : {planes.shape}（dtype={planes.dtype}）")
    print(f"合法著法數     : {board.legal_moves.count()}")
    print(f"legal_mask True: {int(mask.sum())}")
    print(f"著法總維度     : {NUM_MOVES}")

    print("\n前 5 個合法著法的 index round-trip：")
    for move in list(board.legal_moves)[:5]:
        idx = encode_move(move, board.turn)
        back = decode_move(idx, board)
        ok = "OK" if back == move else "不一致！"
        print(f"  {move.uci():6s} → index {idx:5d} → {back.uci():6s}  {ok}")


if __name__ == "__main__":
    main()
