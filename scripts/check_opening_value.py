"""檢查 value head 在開局階段的評分是不是系統性偏高。

    python scripts/check_opening_value.py
    python scripts/check_opening_value.py --lines 100 --depth 12

--------------------------------------------------------------------------
在查什麼
--------------------------------------------------------------------------

value head 對**起始盤面**給出 +0.458（換算成 +2.49 兵），高得離譜。
問題是這個偏差只發生在起始盤面，還是整個開局階段都有？

有一個很乾淨的分界可以用來檢驗：`preprocess.py` 的取樣規則是
`range(skip_opening_plies, ...)`，預設 8 —— 也就是說

    ply 0–7  從來沒有進過訓練集（分佈外）
    ply 8+   有進訓練集（分佈內）

如果偏差在 ply 8 前後出現明顯落差，那成因就是**分佈外外插**：
網路在沒見過的局面上外插出無意義的高分。
如果整個開局都偏高、ply 8 沒有轉折，那就是別的原因。

這個結果會影響 MCTS：如果整個開局階段的 value 都不可信，
搜尋在開局的每個葉節點都拿到錯的評估，前十幾步的著法品質會受影響。

--------------------------------------------------------------------------
視角約定
--------------------------------------------------------------------------

value head 輸出的是**當前走棋方**視角。這支程式一律轉成**白方視角**再比較，
因為 Stockfish 的 `score.white()` 也是白方視角。轉換只發生在 `_white_cp()`。
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

# scripts/ 不是套件，直接執行時 `import src.xxx` 會找不到（跟其他 scripts 一致的做法）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess  # noqa: E402
import chess.engine  # noqa: E402
import chess.pgn  # noqa: E402
import torch  # noqa: E402

from src.config import add_common_args, load_config  # noqa: E402
from src.encoding import encode_board  # noqa: E402
from src.model import ChessNet, resolve_device  # noqa: E402
from src.move_info import value_to_cp  # noqa: E402

DEFAULT_OPENINGS = "data/openings.pgn"
DEFAULT_CHECKPOINT = "models/best.pt"
# 開局書每條 8 半步，要看到 ply 8 之後的對照就得再往前走幾步
MAX_PLY = 12
# Stockfish 的分數夾在這個範圍，避免將死分數（±30000）把平均值拉爆
CP_CLAMP = 1000


@torch.no_grad()
def model_value_white(model: ChessNet, device: torch.device, board: chess.Board) -> float:
    """value head 對這個盤面的直接輸出，**轉成白方視角**。

    Args:
        model: 網路。
        device: 裝置。
        board: 盤面。

    Returns:
        -1 ~ 1，正值代表白方佔優。
    """
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    _, value = model(x)
    v = float(value.item())            # 當前走棋方視角
    return v if board.turn == chess.WHITE else -v


def stockfish_cp_white(
    engine: chess.engine.SimpleEngine, board: chess.Board, depth: int
) -> int:
    """Stockfish 的評分，**白方視角**的 centipawn，夾在 ±CP_CLAMP。"""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    cp = score.score(mate_score=100000)
    return max(-CP_CLAMP, min(CP_CLAMP, int(cp)))


def load_opening_lines(path: Path, count: int, seed: int) -> list[list[chess.Move]]:
    """從開局書隨機取 count 條著法序列。"""
    if not path.exists():
        raise SystemExit(
            f"找不到開局書 {path}\n下一步：python scripts/make_openings.py"
        )
    games: list[list[chess.Move]] = []
    stream = io.StringIO(path.read_text(encoding="utf-8"))
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games.append(list(game.mainline_moves()))
    rng = random.Random(seed)
    rng.shuffle(games)
    return games[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="檢查開局階段 value head 的偏差")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--openings", type=str, default=DEFAULT_OPENINGS)
    parser.add_argument("--lines", type=int, default=60, help="取樣幾條開局")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish 搜尋深度")
    parser.add_argument("--max-ply", type=int, default=MAX_PLY)
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model, _ = ChessNet.from_checkpoint(ckpt_path, device=device)
    model.eval()

    engine_path = cfg.resolve_path(cfg.eval.stockfish_path)
    if not engine_path.exists():
        raise SystemExit(
            f"找不到 Stockfish：{engine_path}\n"
            f"下一步：到 https://stockfishchess.org/download/ 下載後放到這個路徑"
        )

    lines = load_opening_lines(cfg.resolve_path(args.openings), args.lines, cfg.seed)
    skip = cfg.data.skip_opening_plies

    print(f"模型      : {ckpt_path.name}")
    print(f"開局條數  : {len(lines)}")
    print(f"Stockfish : depth {args.depth}")
    print(f"訓練集邊界: ply < {skip} 從未進過訓練集（分佈外）\n")

    # 每個 ply 累積 (模型 cp, Stockfish cp)
    per_ply: dict[int, list[tuple[int, int]]] = {p: [] for p in range(args.max_ply + 1)}

    with chess.engine.SimpleEngine.popen_uci(str(engine_path), timeout=30.0) as engine:
        for index, moves in enumerate(lines, 1):
            board = chess.Board()
            for ply in range(args.max_ply + 1):
                if board.is_game_over():
                    break
                model_cp = value_to_cp(model_value_white(model, device, board))
                sf_cp = stockfish_cp_white(engine, board, args.depth)
                per_ply[ply].append((model_cp, sf_cp))

                if ply < len(moves):
                    board.push(moves[ply])            # 開局書的著法
                else:
                    # 書走完了，用 Stockfish 續走，維持是「真實會出現的局面」
                    board.push(engine.play(board, chess.engine.Limit(depth=args.depth)).move)
            if index % 10 == 0:
                print(f"  已處理 {index}/{len(lines)} 條")

    print(f"\n{'=' * 74}")
    print("開局各 ply 的評分對照（都是白方視角的 centipawn）")
    print(f"{'=' * 74}")
    print(f"{'ply':>4}{'樣本':>6}{'模型':>9}{'Stockfish':>11}{'偏差':>9}{'絕對誤差':>10}  {'訓練集'}")

    rows: list[tuple[int, float, float, float]] = []
    for ply in range(args.max_ply + 1):
        pairs = per_ply[ply]
        if not pairs:
            continue
        model_mean = sum(m for m, _ in pairs) / len(pairs)
        sf_mean = sum(s for _, s in pairs) / len(pairs)
        bias = model_mean - sf_mean
        abs_err = sum(abs(m - s) for m, s in pairs) / len(pairs)
        rows.append((ply, model_mean, sf_mean, bias))
        marker = "分佈外" if ply < skip else "分佈內"
        print(
            f"{ply:>4}{len(pairs):>6}{model_mean:>9.0f}{sf_mean:>11.0f}"
            f"{bias:>+9.0f}{abs_err:>10.0f}  {marker}"
        )

    # 分界線前後的平均偏差，這是判斷成因的關鍵
    outside = [b for p, _, _, b in rows if p < skip]
    inside = [b for p, _, _, b in rows if p >= skip]
    print(f"\n{'=' * 74}")
    if outside and inside:
        out_mean = sum(outside) / len(outside)
        in_mean = sum(inside) / len(inside)
        print(f"分佈外（ply 0–{skip - 1}）平均偏差: {out_mean:+.0f} cp")
        print(f"分佈內（ply {skip}+）  平均偏差: {in_mean:+.0f} cp")
        print(f"落差: {out_mean - in_mean:+.0f} cp")

        start_bias = rows[0][3]
        rest_outside = [b for p, _, _, b in rows if 0 < p < skip]
        rest_mean = sum(rest_outside) / len(rest_outside) if rest_outside else 0.0
        print(f"\n起始盤面單獨的偏差: {start_bias:+.0f} cp")
        print(f"其餘分佈外 ply 的平均偏差: {rest_mean:+.0f} cp")

        print("\n判讀：")
        if abs(out_mean - in_mean) > 50 and abs(rest_mean) > 50:
            print("  整個分佈外區段都系統性偏高，ply 邊界前後有明顯落差")
            print("  → 成因是**分佈外外插**，不是單一盤面的記憶")
            print("  → MCTS 在開局前幾步的葉節點評估都不可信，會影響開局著法品質")
        elif abs(start_bias) > 50 and abs(rest_mean) < 50:
            print("  只有起始盤面偏高，其餘開局盤面正常")
            print("  → 成因侷限在單一盤面，對 MCTS 的實際影響很小")
        else:
            print("  分佈外與分佈內的偏差沒有明顯差別")
            print("  → 開局偏差不是分佈外造成的，要另找原因")


if __name__ == "__main__":
    main()
