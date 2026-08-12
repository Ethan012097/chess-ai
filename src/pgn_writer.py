"""把對局寫成標準 PGN，附上 lichess 認得的評分註解。

所有對局（評估用、人機對弈、自我對弈）一律走這裡，這是後續分析與展示的基礎。

每一步會寫成兩段註解：

    1. e4 {[%eval 0.08]} {policy: e4 .41 d4 .22 Nf3 .11 c4 .08 e3 .04 | 0.03s}

- 第一段 `[%eval x.xx]` 是 **lichess 認得的標準格式**，單位是兵值、**白方視角**。
  把 PGN 貼到 https://lichess.org/paste 就會自動畫出評分曲線與分析板。
- 第二段是我們自己的除錯資訊，lichess 會直接忽略。

用法：
    python -m src.pgn_writer --demo      # 產生一個範例 PGN 檢查格式
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import chess
import chess.pgn

from src.move_info import MoveInfo, value_to_cp

# 註解裡要列幾個候選著法。MoveInfo 存 8 個，但全部印出來太長，預設印 5 個。
COMMENT_POLICY_COUNT = 5

# PGN 一定要有的標頭。ModelCheckpoint 是自訂欄位——沒有它，三個月後會分不清
# 這局是哪一版模型下的。
REQUIRED_HEADERS = (
    "Event",
    "Site",
    "Date",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "ModelCheckpoint",
)


def format_eval_comment(info: MoveInfo) -> str:
    """產生 lichess 認得的 `[%eval x.xx]` 註解。

    **視角轉換就發生在這裡**：`info.value` 是當前走棋方視角，
    `info.value_white()` 轉成白方視角，再換算成兵值（centipawn / 100）。
    黑方走棋時取負號這件事由 `value_white()` 處理。

    Args:
        info: 一步棋的紀錄。

    Returns:
        例如 `[%eval 0.08]`（白方領先 0.08 個兵）。
    """
    cp_white = value_to_cp(info.value_white())
    return f"[%eval {cp_white / 100:.2f}]"


def format_policy_comment(info: MoveInfo, count: int = COMMENT_POLICY_COUNT) -> str:
    """產生自訂的 policy 註解（lichess 會忽略）。

    `policy_top` 存的是 UCI，這裡轉成 SAN 比較好讀。轉換需要「走這步之前」的
    盤面，所以用 `info.fen_before` 重建。

    Args:
        info: 一步棋的紀錄。
        count: 要列幾個候選著法。

    Returns:
        例如 `policy: e4 .41 d4 .22 Nf3 .11 | 0.03s`。
    """
    parts: list[str] = []
    board = chess.Board(info.fen_before) if info.fen_before else None
    for uci, prob in info.policy_top[:count]:
        label = uci
        if board is not None:
            try:
                label = board.san(chess.Move.from_uci(uci))
            except (ValueError, AssertionError):
                # 著法在這個盤面上不合法（理論上不會發生），退回顯示 UCI
                label = uci
        # .41 這種寫法比 0.41 省字，PGN 註解越短越好讀
        parts.append(f"{label} {prob:.2f}".replace("0.", "."))

    text = "policy: " + " ".join(parts)
    if info.visits:
        # MCTS 啟用時附上訪問次數。policy 是「直覺想走哪」，visits 是
        # 「想過之後認為哪裡值得」，兩者意義不同，所以分開列。
        top_visits = sorted(info.visits.items(), key=lambda kv: kv[1], reverse=True)
        visit_text = " ".join(f"{u}:{n}" for u, n in top_visits[:count])
        text += f" | visits: {visit_text}"
    return f"{text} | {info.elapsed_ms / 1000:.2f}s"


def build_game(
    board: chess.Board,
    move_infos: list[MoveInfo],
    headers: dict[str, str],
) -> chess.pgn.Game:
    """把對局組成 `chess.pgn.Game`（`write_game` 與測試共用）。

    Args:
        board: **對局結束後**的盤面（用來取 Result 與驗證步數）。
        move_infos: 每一步的紀錄，順序與實際對局相同。
        headers: PGN 標頭，缺少的必填欄位會補預設值。

    Returns:
        可以直接 `str()` 成 PGN 文字的 Game 物件。
    """
    game = chess.pgn.Game()

    defaults = {
        "Event": "chess-ai game",
        "Site": "local",
        "Date": date.today().strftime("%Y.%m.%d"),
        "White": "?",
        "Black": "?",
        "Result": board.result(claim_draw=True),
        "WhiteElo": "?",
        "BlackElo": "?",
        "ModelCheckpoint": "?",
    }
    for key in REQUIRED_HEADERS:
        game.headers[key] = str(headers.get(key, defaults[key]))
    # 其餘自訂標頭照收
    for key, value in headers.items():
        if key not in REQUIRED_HEADERS:
            game.headers[key] = str(value)

    node: chess.pgn.GameNode = game
    for info in move_infos:
        node = node.add_variation(info.move)
        # 兩段註解中間留一個空格，lichess 只會解析 [%eval ...] 那一段
        node.comment = f"{format_eval_comment(info)} {format_policy_comment(info)}"

    return game


def write_game(
    board: chess.Board,
    move_infos: list[MoveInfo],
    headers: dict[str, str],
    path: Path,
) -> None:
    """把一局棋寫成 PGN 檔。

    Args:
        board: 對局結束後的盤面。
        move_infos: 每一步的紀錄。
        headers: PGN 標頭（見 REQUIRED_HEADERS）。
        path: 輸出路徑，父資料夾不存在會自動建立。
    """
    game = build_game(board, move_infos, headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 一律用 UTF-8 寫檔，不要依賴 Windows 的 cp950 locale
    with open(path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)


def append_game(
    board: chess.Board,
    move_infos: list[MoveInfo],
    headers: dict[str, str],
    path: Path,
) -> None:
    """把一局棋附加到既有的 PGN 檔尾端（多局放同一個檔時用）。"""
    game = build_game(board, move_infos, headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(game))
        f.write("\n\n")


def main() -> None:
    """`python -m src.pgn_writer --demo` 產生一個範例 PGN，用來檢查格式。"""
    parser = argparse.ArgumentParser(description="PGN 輸出（含 lichess 評分註解）")
    parser.add_argument("--demo", action="store_true", help="產生一個手工範例 PGN")
    parser.add_argument(
        "--out", type=str, default="logs/demo/sample.pgn", help="輸出路徑"
    )
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        print("\n這支模組主要是給別的程式呼叫的。要看範例輸出請加 --demo。")
        return

    # 手工造一局極短的棋，數值是假的，只為了檢查格式與視角
    board = chess.Board()
    infos: list[MoveInfo] = []
    # 白方走 e4，value=+0.08（白方視角，因為現在是白方走）
    infos.append(
        MoveInfo(
            move=chess.Move.from_uci("e2e4"),
            san="e4",
            value=0.08,
            policy_top=[("e2e4", 0.41), ("d2d4", 0.22), ("g1f3", 0.11)],
            elapsed_ms=30,
            fen_before=board.fen(),
        )
    )
    board.push(chess.Move.from_uci("e2e4"))
    # 黑方走 e5，value=+0.05（黑方視角）→ 寫進 PGN 應變成 -0.05（白方視角）
    infos.append(
        MoveInfo(
            move=chess.Move.from_uci("e7e5"),
            san="e5",
            value=0.05,
            policy_top=[("e7e5", 0.38), ("c7c5", 0.30)],
            elapsed_ms=28,
            fen_before=board.fen(),
        )
    )
    board.push(chess.Move.from_uci("e7e5"))

    path = Path(args.out)
    write_game(
        board,
        infos,
        {
            "Event": "format demo",
            "White": "MyNet",
            "Black": "MyNet",
            "Result": "*",
            "ModelCheckpoint": "models/best.pt",
        },
        path,
    )
    print(f"已寫出 {path}\n")
    print(path.read_text(encoding="utf-8"))
    print("視角檢查：黑方走棋時 value=+0.05（黑方視角），[%eval] 應為負值（白方視角）。")


if __name__ == "__main__":
    main()
