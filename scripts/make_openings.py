"""從 PGN 抽開局書，給引擎對戰當起始位置用。

用法（在專案根目錄執行）：
    python scripts/make_openings.py --input "data/raw/*.pgn"
    python scripts/make_openings.py --input "data/raw/*.pgn" --min-count 20

為什麼需要開局書：自動對戰若每局都從起始盤面開始，同一個確定性引擎會下出
一模一樣的棋，統計毫無意義（規格 §3.1）。

規則：
  - 取每盤棋的前 8 個半步
  - 該開局在原始資料中至少要出現 `--min-count` 次（預設 50），避免抽到冷僻變化
  - 去重後隨機取 `--count` 個（預設 2000）
  - 輸出 `data/openings.pgn`，格式給 cutechess-cli 的 `-openings format=pgn` 用

效能：這支**不用** `chess.pgn.read_game`。那支會把整盤棋建成樹，
但我們只要前 8 個半步，用它等於慢上十幾倍。改成直接掃文字、
只解析前幾個 SAN token。
"""

from __future__ import annotations

import argparse
import glob
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PLIES = 8
DEFAULT_COUNT = 2000
DEFAULT_MIN_COUNT = 50
DEFAULT_OUTPUT = "data/openings.pgn"

# SAN token：易位、帶棋子代號的著法、兵的著法（含吃子與升變）。
# 後面的 [+#]? 吃掉將軍 / 將死記號，[!?]* 吃掉註解記號。
SAN_TOKEN = re.compile(
    r"(O-O-O|O-O"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?"
    r"|[a-h][1-8](?:=[QRBN])?"
    r")[+#]?[!?]*"
)


def iter_movetexts(path: Path) -> Iterator[str]:
    """逐局吐出 PGN 的著法段落（不含標頭）。

    PGN 的結構是「標頭區塊 → 空行 → 著法區塊 → 空行」。這裡只做最低限度的
    切分，不驗證格式——壞掉的那一局頂多被後面的 SAN 解析擋掉。

    Args:
        path: PGN 檔案。

    Yields:
        每一局的著法文字。
    """
    buffer: list[str] = []
    in_moves = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                # 新的一局開始了，把上一局的著法吐出去
                if in_moves and buffer:
                    yield " ".join(buffer)
                    buffer = []
                in_moves = False
                continue
            if not stripped:
                continue
            in_moves = True
            buffer.append(stripped)
    if buffer:
        yield " ".join(buffer)


def extract_opening(movetext: str, plies: int) -> tuple[str, ...] | None:
    """從著法段落取出前 N 個半步。

    Args:
        movetext: 一局的著法文字。
        plies: 要取幾個半步。

    Returns:
        UCI 著法組成的 tuple；步數不足或有非法著法時回傳 None。
        回傳 UCI 而不是 SAN，因為 UCI 沒有歧義，拿來當 dict 的 key 才安全。
    """
    board = chess.Board()
    moves: list[str] = []
    for match in SAN_TOKEN.finditer(movetext):
        if len(moves) >= plies:
            break
        try:
            move = board.parse_san(match.group(1))
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            # 掃到的不是真的著法（例如結果標記或註解殘留），整局放棄
            return None
        board.push(move)
        moves.append(move.uci())

    return tuple(moves) if len(moves) == plies else None


def collect_openings(
    paths: list[Path], plies: int, max_games: int | None = None
) -> Counter[tuple[str, ...]]:
    """掃過所有 PGN，統計每個開局出現幾次。

    Args:
        paths: PGN 檔案清單。
        plies: 開局長度（半步）。
        max_games: 最多讀幾局（除錯用）。

    Returns:
        {開局(UCI tuple): 出現次數}
    """
    counter: Counter[tuple[str, ...]] = Counter()
    games_read = 0

    for path in paths:
        print(f"[讀取] {path}")
        bar = tqdm(desc=path.name, unit="局")
        for movetext in iter_movetexts(path):
            games_read += 1
            bar.update(1)
            opening = extract_opening(movetext, plies)
            if opening is not None:
                counter[opening] += 1
            if max_games is not None and games_read >= max_games:
                break
        bar.close()
        if max_games is not None and games_read >= max_games:
            break

    print(f"讀取 {games_read:,} 局，取得 {len(counter):,} 種不同開局")
    return counter


def write_openings(
    openings: list[tuple[str, ...]], path: Path, counts: Counter[tuple[str, ...]]
) -> None:
    """把開局寫成 PGN。

    cutechess-cli 用 `-openings file=... format=pgn` 讀這個檔，
    每一局的著法就是一個開局位置。

    Args:
        openings: 要寫出的開局清單。
        path: 輸出路徑。
        counts: 出現次數，寫進標頭方便事後查看。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        for i, opening in enumerate(openings, start=1):
            game = chess.pgn.Game()
            game.headers["Event"] = "opening book"
            game.headers["Site"] = "local"
            game.headers["Round"] = str(i)
            game.headers["White"] = "?"
            game.headers["Black"] = "?"
            game.headers["Result"] = "*"
            game.headers["Occurrences"] = str(counts[opening])

            node: chess.pgn.GameNode = game
            for uci in opening:
                node = node.add_variation(chess.Move.from_uci(uci))
            game.accept(exporter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="從 PGN 抽開局書給引擎對戰用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["data/raw/*.pgn"],
        help='PGN 檔案或萬用字元（PowerShell 不會自己展開 *，記得加引號）',
    )
    parser.add_argument("--plies", type=int, default=DEFAULT_PLIES, help="開局長度（半步）")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="要輸出幾個開局")
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help="開局至少要在原始資料出現幾次（避免冷僻變化）",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="輸出路徑")
    parser.add_argument("--max-games", type=int, default=None, help="最多讀幾局（除錯用）")
    parser.add_argument("--seed", type=int, default=42, help="亂數種子")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        paths.extend(Path(m) for m in matched)
    if not paths:
        raise SystemExit(
            f"找不到符合的 PGN：{args.input}\n"
            f"下一步：先下載棋譜\n"
            f"  python scripts/download_data.py --source elite --month 2025-11"
        )

    counter = collect_openings(paths, args.plies, args.max_games)
    if not counter:
        raise SystemExit("沒有抽到任何開局，請確認 PGN 內容是否正常。")

    frequent = [op for op, n in counter.items() if n >= args.min_count]
    print(f"出現 >= {args.min_count} 次的開局：{len(frequent):,} 種")

    if len(frequent) < args.count:
        print(
            f"\n[提醒] 符合條件的開局只有 {len(frequent):,} 種，少於要求的 {args.count:,} 個。\n"
            f"        8 個半步已經相當深，能重複出現 {args.min_count} 次的變化本來就不多。\n"
            f"        可以擇一：\n"
            f"          - 降低門檻：--min-count {max(args.min_count // 5, 2)}\n"
            f"          - 縮短開局：--plies 6\n"
            f"          - 多讀幾個月的棋譜\n"
        )

    rng = random.Random(args.seed)
    chosen = frequent if len(frequent) <= args.count else rng.sample(frequent, args.count)
    # 依出現次數由高到低排序，讓檔案前面是最主流的開局
    chosen.sort(key=lambda op: counter[op], reverse=True)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    write_openings(chosen, out_path, counter)

    print(f"\n已寫出 {len(chosen):,} 個開局到 {out_path}")
    if chosen:
        board = chess.Board()
        sans = []
        for uci in chosen[0]:
            move = chess.Move.from_uci(uci)
            sans.append(board.san(move))
            board.push(move)
        print(f"最常見的開局（{counter[chosen[0]]:,} 次）：{' '.join(sans)}")

    print("\n下一步：")
    print("  python -m src.evaluate --mode tournament")


if __name__ == "__main__":
    main()
