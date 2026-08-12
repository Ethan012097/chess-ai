"""讓模型自我對弈幾局，存成 PGN，用來檢查棋風與評分曲線。

用法（在專案根目錄執行）：
    python scripts/make_demo_games.py                      # 預設 10 局
    python scripts/make_demo_games.py --games 3 --temperature 0.5
    python scripts/make_demo_games.py --checkpoint models/epoch_12.pt

產出 `logs/demo/game_{n}.pgn`。把任何一個檔案貼到 https://lichess.org/paste
就能逐步播放並看到評分曲線 —— 這是 value 視角有沒有搞反的實地測試，
比單元測試更容易抓到問題（規格 §1.6）。

為什麼 temperature 預設 0.3 而不是 0：temperature=0 是純 argmax，模型是確定性的，
10 局會下出一模一樣的棋，看不出任何東西。0.3 讓它偶爾偏離最佳著法，
但又不會亂走（機率取 1/0.3 次方後，主要著法還是壓倒性地高）。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_common_args, load_config  # noqa: E402
from src.model import ChessNet, resolve_device  # noqa: E402
from src.move_info import MoveInfo, make_move_info  # noqa: E402
from src.pgn_writer import write_game  # noqa: E402
from src.search.greedy import GreedySearcher  # noqa: E402

DEFAULT_GAMES = 10
DEFAULT_TEMPERATURE = 0.3
# 開局隨機步數（data/openings.pgn 還沒做出來時的退路，見規格 §1.5）
FALLBACK_OPENING_PLIES = 4
# 單局步數上限，避免沒完沒了的和棋卡住整批產生
MAX_PLIES = 300
OPENINGS_FILE = "data/openings.pgn"


def load_openings(path: Path, limit: int = 2000) -> list[list[chess.Move]]:
    """從 openings.pgn 讀開局著法序列。

    Args:
        path: openings.pgn 路徑。
        limit: 最多讀幾個開局。

    Returns:
        每個元素是一串著法；檔案不存在時回傳空 list（呼叫端會改用隨機開局）。
    """
    if not path.exists():
        return []
    openings: list[list[chess.Move]] = []
    with open(path, encoding="utf-8") as f:
        while len(openings) < limit:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if moves:
                openings.append(moves)
    return openings


def apply_opening(
    board: chess.Board,
    openings: list[list[chess.Move]],
    rng: random.Random,
) -> None:
    """在盤面上套用一個開局（就地修改 board）。

    有 openings.pgn 就從裡面隨機挑一個；沒有就隨機走前幾個半步。

    Args:
        board: 會被就地修改。
        openings: `load_openings` 的輸出，空 list 表示改用隨機開局。
        rng: 亂數產生器。
    """
    if openings:
        for move in rng.choice(openings):
            if move in board.legal_moves:
                board.push(move)
            else:
                break
        return

    for _ in range(FALLBACK_OPENING_PLIES):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            return
        board.push(rng.choice(moves))


def play_one_game(
    searcher: GreedySearcher,
    openings: list[list[chess.Move]],
    rng: random.Random,
) -> tuple[chess.Board, list[MoveInfo], int]:
    """自我對弈一局。

    Args:
        searcher: 兩邊都用同一個 searcher（自我對弈）。
        openings: 開局來源。
        rng: 亂數產生器。

    Returns:
        (結束後的盤面, 每一步的 MoveInfo, 開局用掉的半步數)
        開局那幾步沒有網路評估，所以不會出現在 MoveInfo 裡 —— 但它們在 board
        的 move_stack 裡，寫 PGN 時要一起帶上，見 `build_move_infos_with_opening`。
    """
    board = chess.Board()
    apply_opening(board, openings, rng)
    opening_plies = len(board.move_stack)

    move_infos: list[MoveInfo] = []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
        start = time.perf_counter()
        # analyse 一次前向就同時拿到 value 與 policy
        value, probs = searcher.analyse(board)
        move = searcher.select_move(board)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        move_infos.append(make_move_info(board, move, value, probs, elapsed_ms))
        board.push(move)

    return board, move_infos, opening_plies


def build_opening_infos(board: chess.Board, opening_plies: int) -> list[MoveInfo]:
    """把開局那幾個隨機半步補成 MoveInfo，這樣 PGN 才是完整一局。

    這幾步是隨機走的、沒有經過網路，所以 value 記 0、policy_top 留空。
    它們只有前幾個半步，對評分曲線的影響有限。

    Args:
        board: 結束後的盤面（用它的 move_stack 重播開局）。
        opening_plies: 開局用掉幾個半步。

    Returns:
        開局那幾步的 MoveInfo。
    """
    infos: list[MoveInfo] = []
    replay = chess.Board()
    for move in board.move_stack[:opening_plies]:
        infos.append(
            MoveInfo(
                move=move,
                san=replay.san(move),
                value=0.0,
                policy_top=[],
                elapsed_ms=0,
                fen_before=replay.fen(),
            )
        )
        replay.push(move)
    return infos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="讓模型自我對弈並存成 PGN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="要下幾局")
    parser.add_argument(
        "--checkpoint", type=str, default="models/best.pt", help="模型 checkpoint"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="取樣溫度。0 = 每局都一樣，建議 0.3",
    )
    parser.add_argument(
        "--out-dir", type=str, default="logs/demo", help="輸出資料夾"
    )
    parser.add_argument("--seed", type=int, default=None, help="亂數種子")
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    model, checkpoint = ChessNet.from_checkpoint(checkpoint_path, device=device)
    searcher = GreedySearcher(model, device, cfg, temperature=args.temperature)

    openings = load_openings(PROJECT_ROOT / OPENINGS_FILE)
    if openings:
        print(f"開局來源：{OPENINGS_FILE}（{len(openings)} 個）")
    else:
        print(
            f"找不到 {OPENINGS_FILE}，改用隨機開局（前 {FALLBACK_OPENING_PLIES} 個半步）。\n"
            f"要產生開局書請跑（第 3 節）：python scripts/make_openings.py"
        )

    rng = random.Random(args.seed if args.seed is not None else cfg.seed)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    print(f"模型   ：{checkpoint_path.name}（epoch {checkpoint.get('epoch', '?')}）")
    print(f"溫度   ：{args.temperature}")
    print(f"輸出   ：{out_dir}\n")

    results: dict[str, int] = {}
    for i in tqdm(range(1, args.games + 1), desc="自我對弈", unit="局"):
        board, move_infos, opening_plies = play_one_game(searcher, openings, rng)
        all_infos = build_opening_infos(board, opening_plies) + move_infos

        result = board.result(claim_draw=True)
        results[result] = results.get(result, 0) + 1

        path = out_dir / f"game_{i}.pgn"
        write_game(
            board,
            all_infos,
            {
                "Event": "self-play demo",
                "White": "MyNet",
                "Black": "MyNet",
                "Result": result,
                "ModelCheckpoint": checkpoint_path.name,
                "Temperature": str(args.temperature),
                "PlyCount": str(len(board.move_stack)),
            },
            path,
        )

    print(f"\n產出 {args.games} 局到 {out_dir}")
    print(f"結果分佈：{results}")
    print("\n驗收（規格 §1.6）：")
    print(f"  1. 開啟 https://lichess.org/paste")
    print(f"  2. 貼上 {out_dir / 'game_1.pgn'} 的內容")
    print(f"  3. 確認：能逐步播放、評分曲線有畫出來、")
    print(f"     且曲線方向合理（白方優勢時在上方）")


if __name__ == "__main__":
    main()
