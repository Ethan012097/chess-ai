"""自我對弈訓練迴圈（規格 §7）。

    python -m src.selfplay --iterations 1              # 跑一個完整 iteration
    python -m src.selfplay --games 20 --no-train       # 只產生對局，不訓練
    python -m src.selfplay --smoke-test                # 4 局 + 50 步訓練，確認整條路通

--------------------------------------------------------------------------
先講清楚期望值（規格 §7.1）
--------------------------------------------------------------------------

AlphaZero 從零開始訓練用了 5000 顆 TPU。**你有一張顯卡。**
所以這裡不是從零開始，而是**從監督式模型出發繼續進化**。
合理預期：跑一到兩週，棋力提升幾十到一兩百 Elo，然後進入緩慢爬升。

若跑了三天完全沒進步，先回頭檢查視角測試（`pytest tests/test_mcts.py`），
而不是加大模型。

--------------------------------------------------------------------------
一個 iteration 的四個步驟（規格 §7.3）
--------------------------------------------------------------------------

1. **產生對局**：用目前的 `best.pt` 自我對弈 N 局，每步 400 次模擬
2. **寫入 replay buffer**：用稀疏格式存成 `data/selfplay/iter_XXXX.npy`
3. **訓練**：從 buffer 抽樣訓練 M 步，`soft_targets: true`
4. **把關**：SPRT 對打舊版，通過才更新 `best.pt`

--------------------------------------------------------------------------
視角約定
--------------------------------------------------------------------------

- MCTS 的 value、`value_target` 欄位：**當前走棋方**視角
- 一盤棋結束後回填 value target 時，白方走的盤面填白方的分數、
  黑方走的盤面填黑方的分數 —— 也就是說**同一盤棋裡相鄰兩步的 target 互為相反數**
  （和局除外，都是 0）
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
from tqdm import tqdm

from src.config import PROJECT_ROOT, Config, add_common_args, load_config
from src.encoding import NUM_MOVES, encode_board_compact
from src.model import ChessNet, resolve_device
from src.preprocess import (
    SELFPLAY_DTYPE,
    SPARSE_POLICY_K,
    VALUE_QUANT_SCALE,
    encode_sparse_policy,
)
from src.search.mcts import MCTSSearcher

# 自我對弈的資料放這裡，一個 iteration 一個檔
SELFPLAY_DIR = "data/selfplay"
LOG_FILE_NAME = "selfplay_log.csv"

# 每步的模擬次數。比正式對局的 800 少，換取資料量（規格 §7.3）。
DEFAULT_SIMULATIONS = 400
# 批次大小 32 是實測的甜蜜點：再大反而變慢，因為 virtual loss 會讓同一批的路徑
# 大量重疊，多做的樹走訪抵銷掉 GPU 那點好處。
LEAF_BATCH_SIZE = 32

# 連續這麼多步 root value 低於門檻就認輸（規格 §7.5）
RESIGN_CONSECUTIVE = 10
# 保留這個比例的對局**不啟用認輸**，用來統計誤判率（應低於 5 %）
RESIGN_AUDIT_RATIO = 0.1
# 超過這個步數就判和，避免罕見的無限拉鋸吃掉整個 iteration
MAX_GAME_PLIES = 400
# 每下這麼多局就把目前累積的資料落地一次。
# **不做這件事的代價很實際**：自我對弈一代要好幾小時，中途被中斷
#（關機、睡眠、session 結束）就什麼都不留。實測連續兩次中斷，
# 各損失數十局的計算，replay buffer 完全是空的。
SAVE_EVERY_GAMES = 25

# replay buffer 保留最近幾個 iteration（滑動視窗，避免被早期的爛資料拖住）
BUFFER_ITERATIONS = 20
# 抽樣時較新的 iteration 權重較高：權重 = decay^(距今幾代)
BUFFER_RECENCY_DECAY = 0.9

# 被截斷的機率質量平均超過這個值就代表 SPARSE_POLICY_K 太小（規格 §7.2）
TRUNCATION_WARN = 0.02
# 和局比例超過這個值代表模型過度保守（規格 §7.6）
DRAW_RATE_WARN = 0.70

CSV_COLUMNS = [
    "iteration",
    "games",
    "positions",
    "avg_plies",
    "draw_rate",
    "resign_rate",
    "false_resign_rate",
    "avg_truncated_mass",
    "train_steps",
    "policy_loss",
    "value_loss",
    "sprt_result",
    "elo",
    "accepted",
    "seconds",
]


@dataclass
class GameResult:
    """一盤自我對弈的產物。

    Attributes:
        rows: 可以直接塞進 SELFPLAY_DTYPE 的 tuple 串。
        plies: 總步數。
        result: 白方視角的分數（+1 白勝、0 和、-1 黑勝）。
        resigned: 這盤是不是以認輸收場。
        audit_wrong_resign: 這盤關掉了認輸，而且「如果有開就會認錯」。
            只有審計局（10 %）會設成 True/False，其餘為 None。
        truncated_mass: 每個盤面被截斷掉的機率質量平均值。
    """

    rows: list[tuple] = field(default_factory=list)
    plies: int = 0
    result: int = 0
    resigned: bool = False
    audit_wrong_resign: bool | None = None
    truncated_mass: float = 0.0


# --- 一盤棋 -----------------------------------------------------------------


def play_one_game(
    searcher: MCTSSearcher,
    temperature_moves: int,
    resign_threshold: float,
    allow_resign: bool,
    rng: np.random.Generator,
) -> GameResult:
    """自我對弈一盤，回傳這盤的所有訓練樣本。

    Args:
        searcher: MCTS（`add_noise=True`，自我對弈一定要加 Dirichlet noise）。
        temperature_moves: 前幾步用 temperature=1 抽樣（製造變化），之後用 argmax。
        resign_threshold: root value 低於這個值算「很糟」，預設 -0.9。
        allow_resign: False 表示這是審計局，就算很糟也要下完。
        rng: 亂數產生器。

    Returns:
        GameResult。`rows` 裡的 value_target 已經回填成最終結果。

    **policy target 是 π(a) = N(a) / ΣN**，固定用 temperature=1 的分佈，
    跟「實際走哪一步」用的抽樣溫度無關 —— 這是 AlphaZero 的做法：
    訓練目標要保留搜尋的完整資訊，抽樣溫度只是為了讓對局有變化。
    """
    board = chess.Board()
    # (緊湊盤面, top_indices, top_probs, 走這步的是白方嗎)
    pending: list[tuple[tuple, np.ndarray, np.ndarray, bool]] = []
    truncated: list[float] = []

    bad_streak = 0
    resigned_by: chess.Color | None = None
    would_resign_by: chess.Color | None = None

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_GAME_PLIES:
        root = searcher.run_simulations(board)
        visits = {
            index: child.visit_count
            for index, child in root.children.items()
            if child.visit_count > 0
        }
        if not visits:
            break                       # 一次模擬都沒跑成，當作下不下去了

        total_visits = sum(visits.values())
        probs = {index: count / total_visits for index, count in visits.items()}
        top_indices, top_probs, truncated_mass = encode_sparse_policy(probs)
        truncated.append(truncated_mass)

        pieces, castling, ep_square, halfmove = encode_board_compact(board)
        pending.append(
            ((pieces, castling, ep_square, halfmove), top_indices, top_probs, board.turn == chess.WHITE)
        )

        # 認輸判斷：root.q() 是**當前走棋方**視角，低於門檻代表這一方快輸了
        if root.q() < resign_threshold:
            bad_streak += 1
        else:
            bad_streak = 0
        if bad_streak >= RESIGN_CONSECUTIVE:
            if allow_resign:
                resigned_by = board.turn
                break
            if would_resign_by is None:
                would_resign_by = board.turn   # 審計局：記下來但繼續下

        move = _sample_move(root, probs, board.ply() < temperature_moves, rng)
        if move is None:
            break
        board.push(move)

    white_score = _final_white_score(board, resigned_by)

    # 回填 value target：每個盤面填**該盤面走棋方**的分數
    rows: list[tuple] = []
    for (compact, top_indices, top_probs, is_white) in pending:
        score = white_score if is_white else -white_score
        pieces, castling, ep_square, halfmove = compact
        rows.append(
            (
                pieces,
                castling,
                ep_square,
                halfmove,
                top_indices,
                top_probs,
                int(round(score * VALUE_QUANT_SCALE)),
            )
        )

    audit_wrong: bool | None = None
    if not allow_resign and would_resign_by is not None:
        # 「本來會認輸的那一方，最後其實沒有輸」→ 這次認輸是誤判
        resigner_score = white_score if would_resign_by == chess.WHITE else -white_score
        audit_wrong = resigner_score >= 0

    return GameResult(
        rows=rows,
        plies=board.ply(),
        result=white_score,
        resigned=resigned_by is not None,
        audit_wrong_resign=audit_wrong,
        truncated_mass=float(np.mean(truncated)) if truncated else 0.0,
    )


def _sample_move(
    root, probs: dict[int, float], use_temperature: bool, rng: np.random.Generator
) -> chess.Move | None:
    """依訪問次數挑一步。

    Args:
        root: 搜尋完的根節點（`root.moves` 是 index → Move 的對照表）。
        probs: {著法 index: 訪問次數比例}。
        use_temperature: True 用 temperature=1 抽樣，False 取最高。
        rng: 亂數產生器。

    Returns:
        著法；對照表查不到時回 None。
    """
    indices = list(probs)
    if use_temperature:
        weights = np.array([probs[i] for i in indices], dtype=np.float64)
        weights /= weights.sum()
        chosen = indices[int(rng.choice(len(indices), p=weights))]
    else:
        chosen = max(probs, key=probs.get)
    return root.moves.get(chosen)


def _final_white_score(board: chess.Board, resigned_by: chess.Color | None) -> int:
    """算白方視角的最終分數。

    Args:
        board: 結束時的盤面。
        resigned_by: 認輸的一方；None 表示下到自然結束。

    Returns:
        +1 白勝 / 0 和 / -1 黑勝。步數上限截斷時算和局。
    """
    if resigned_by is not None:
        return -1 if resigned_by == chess.WHITE else 1
    if not board.is_game_over(claim_draw=True):
        return 0                      # 撞到 MAX_GAME_PLIES，當和局
    outcome = board.result(claim_draw=True)
    return {"1-0": 1, "0-1": -1}.get(outcome, 0)


# --- 產生一批對局 -----------------------------------------------------------


def generate_games(
    model: ChessNet,
    device: torch.device,
    cfg: Config,
    num_games: int,
    simulations: int,
    seed: int,
    on_progress=None,
) -> tuple[list[tuple], dict[str, float]]:
    """自我對弈 N 局，回傳所有樣本與統計。

    Args:
        model: 目前的模型。
        device: 推論裝置。
        cfg: 設定。
        num_games: 要下幾局。
        simulations: 每步的模擬次數。
        seed: 亂數種子。
        on_progress: 每 SAVE_EVERY_GAMES 局呼叫一次 `on_progress(rows)`，
            用來把中途結果落地。中斷時才不會整代白跑。

    Returns:
        (rows, stats)。rows 可直接寫成 SELFPLAY_DTYPE 陣列。
    """
    mcts_cfg = cfg.mcts or {}
    temperature_moves = int(mcts_cfg.get("temperature_moves", 30))
    resign_threshold = float((cfg.selfplay or {}).get("resign_threshold", -0.9))

    searcher = MCTSSearcher(
        model,
        device,
        cfg,
        simulations=simulations,
        add_noise=True,             # 自我對弈一定要加，不然每局都一樣
        batch_size=LEAF_BATCH_SIZE,
    )
    rng = np.random.default_rng(seed)
    searcher.rng = rng

    rows: list[tuple] = []
    plies: list[int] = []
    draws = resigns = 0
    audits = audit_wrong = 0
    truncated: list[float] = []

    bar = tqdm(range(num_games), desc="自我對弈", unit="局")
    for i in bar:
        allow_resign = rng.random() >= RESIGN_AUDIT_RATIO
        game = play_one_game(
            searcher, temperature_moves, resign_threshold, allow_resign, rng
        )
        rows.extend(game.rows)
        plies.append(game.plies)
        draws += 1 if game.result == 0 else 0
        resigns += 1 if game.resigned else 0
        truncated.append(game.truncated_mass)
        if game.audit_wrong_resign is not None:
            audits += 1
            audit_wrong += 1 if game.audit_wrong_resign else 0
        bar.set_postfix(
            {"盤面": f"{len(rows):,}", "平均步數": f"{np.mean(plies):.0f}",
             "和局": f"{draws / (i + 1) * 100:.0f}%"}
        )
        # 定期落地。中斷的話至少留下已經算完的部分。
        if on_progress is not None and (i + 1) % SAVE_EVERY_GAMES == 0:
            on_progress(rows)
    bar.close()

    stats = {
        "games": num_games,
        "positions": len(rows),
        "avg_plies": float(np.mean(plies)) if plies else 0.0,
        "draw_rate": draws / max(num_games, 1),
        "resign_rate": resigns / max(num_games, 1),
        # 審計局只佔 10 %，局數少的時候這個數字會很跳，看趨勢就好
        "false_resign_rate": audit_wrong / audits if audits else 0.0,
        "avg_truncated_mass": float(np.mean(truncated)) if truncated else 0.0,
    }
    return rows, stats


def save_shard(rows: list[tuple], path: Path) -> None:
    """把樣本寫成一個 .npy（SELFPLAY_DTYPE）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(rows, dtype=SELFPLAY_DTYPE)
    np.save(path, arr)


# --- replay buffer ----------------------------------------------------------


def buffer_files(directory: Path) -> list[Path]:
    """列出 buffer 裡的所有 shard，由舊到新排序。"""
    if not directory.exists():
        return []
    return sorted(directory.glob("iter_*.npy"))


def prune_buffer(directory: Path, keep: int = BUFFER_ITERATIONS) -> list[Path]:
    """只保留最近 `keep` 個 iteration 的資料，更舊的刪掉（規格 §7.4）。

    Returns:
        刪掉的檔案清單。
    """
    files = buffer_files(directory)
    removed = files[:-keep] if len(files) > keep else []
    for path in removed:
        path.unlink()
    return removed


def open_buffer(directory: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """把 buffer 裡所有 shard 開成 memmap，並算好抽樣權重。

    **要在訓練迴圈外面呼叫一次就好。** 每步重開一次 memmap 的話，
    20 個 shard × 2000 步就是四萬次開檔，光是解析 .npy 標頭就會拖垮訓練。

    Args:
        directory: buffer 目錄。

    Returns:
        (arrays, weights)。weights 已正規化，總和為 1。

    Raises:
        FileNotFoundError: buffer 是空的（訊息會告訴你要先跑哪一行）。
    """
    files = buffer_files(directory)
    if not files:
        raise FileNotFoundError(
            f"replay buffer 是空的（{directory}）。\n"
            f"下一步：python -m src.selfplay --games 20 --no-train"
        )

    arrays = [np.load(p, mmap_mode="r") for p in files]
    sizes = np.array([len(a) for a in arrays], dtype=np.float64)
    # 越新的 iteration 權重越高；再乘上該檔的盤面數，
    # 不然盤面少的 shard 裡每一筆被抽中的機率會不成比例地高
    age = np.arange(len(files) - 1, -1, -1, dtype=np.float64)
    weights = (BUFFER_RECENCY_DECAY**age) * sizes
    weights /= weights.sum()
    return arrays, weights


def sample_arrays(
    arrays: list[np.ndarray],
    weights: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """從已開好的 shard 抽樣（規格 §7.4：較新的 iteration 權重較高）。

    Returns:
        SELFPLAY_DTYPE 的陣列（複製出來的，不是 memmap）。
    """
    picks = rng.choice(len(arrays), size=num_samples, p=weights)
    out = np.empty(num_samples, dtype=SELFPLAY_DTYPE)
    for file_index in np.unique(picks):
        slots = np.flatnonzero(picks == file_index)
        source = arrays[int(file_index)]
        indices = rng.integers(0, len(source), size=len(slots))
        out[slots] = source[indices]
    return out


def sample_from_buffer(
    directory: Path, num_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """開 buffer 並抽樣。一次性的用途（檢查資料、測試）用這支就好。"""
    arrays, weights = open_buffer(directory)
    return sample_arrays(arrays, weights, num_samples, rng)


# --- 訓練 -------------------------------------------------------------------


def train_on_buffer(
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    directory: Path,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """從 replay buffer 抽樣訓練若干步。

    刻意不走 `DataLoader`：buffer 是「每步重新抽樣」而不是「跑過一輪」，
    用 DataLoader 反而要多包一層 Sampler。這裡直接抽、直接展開，程式短很多。

    Args:
        model: 要訓練的模型（就地更新）。
        optimizer: 由呼叫端建立並**跨 iteration 保留**，見 `make_optimizer`。
        device: 裝置。
        cfg: 設定。
        directory: buffer 目錄。
        steps: 訓練幾步。
        batch_size: 每步幾筆。
        seed: 亂數種子。

    Returns:
        {"policy_loss", "value_loss"}，最後 20 % 步數的平均。
    """
    from src.dataset import ChessPositionDataset
    from src.train import compute_loss

    rng = np.random.default_rng(seed)
    model.train()

    # memmap 只開一次，不要每步重開（見 open_buffer 的說明）
    arrays, weights = open_buffer(directory)
    policy_losses: list[float] = []
    value_losses: list[float] = []

    bar = tqdm(range(steps), desc="訓練", unit="step")
    for step in bar:
        batch = sample_arrays(arrays, weights, batch_size, rng)

        boards = np.stack([ChessPositionDataset._decode_planes(row) for row in batch])
        policy_target = np.zeros((batch_size, NUM_MOVES), dtype=np.float32)
        for i, row in enumerate(batch):
            mask = row["top_probs"] > 0
            if mask.any():
                policy_target[i, row["top_indices"][mask].astype(np.int64)] = row["top_probs"][mask]
                policy_target[i] /= policy_target[i].sum()
        value_target = batch["value_target"].astype(np.float32) / VALUE_QUANT_SCALE

        x = torch.from_numpy(boards).to(device)
        pt = torch.from_numpy(policy_target).to(device)
        vt = torch.from_numpy(value_target).to(device)

        optimizer.zero_grad(set_to_none=True)
        policy_logits, value = model(x)
        loss, policy_loss, value_loss = compute_loss(
            policy_logits.float(), value.float(), pt, vt, cfg.train.value_weight
        )
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        if step % 20 == 0:
            bar.set_postfix({"p_loss": f"{policy_loss.item():.3f}", "v_loss": f"{value_loss.item():.3f}"})
    bar.close()
    model.eval()

    tail = max(1, steps // 5)
    return {
        "policy_loss": float(np.mean(policy_losses[-tail:])),
        "value_loss": float(np.mean(value_losses[-tail:])),
    }


def make_optimizer(model: ChessNet, cfg: Config, lr: float) -> torch.optim.Optimizer:
    """建立訓練用的 optimizer。

    lr 預設比監督式階段小一到兩個數量級：自我對弈的資料量遠小於人類棋譜，
    用原本的 lr 會直接把既有棋力洗掉。
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)


def load_training_model(
    best_path: Path, candidate_path: Path, device: torch.device, from_best: bool
) -> tuple[ChessNet, dict]:
    """載入「要繼續訓練的那個模型」。

    **這裡有一個容易寫錯的地方。** 直覺會想每個 iteration 都從 `best.pt` 開始訓練，
    但那樣的話：SPRT 把關沒過 → 訓練成果全丟掉 → 下一代從同一個起點、
    用差不多的資料再訓練一次 → 結果也差不多 → **永遠過不了關**。

    正確做法（AlphaGo Zero 的做法）是把兩件事分開：
      - `best.pt`：**產生對局**用的模型，只有通過 SPRT 才更新
      - `selfplay_candidate.pt`：**持續訓練**的模型，每代都在它上面繼續練

    Args:
        best_path: models/best.pt。
        candidate_path: models/selfplay_candidate.pt。
        device: 裝置。
        from_best: True 表示強制從 best.pt 重新開始（`--from-best`）。

    Returns:
        (model, checkpoint)。checkpoint 裡可能有 optimizer_state_dict。
    """
    if not from_best and candidate_path.exists():
        model, ckpt = ChessNet.from_checkpoint(candidate_path, device=device)
        print(f"  接續訓練 {candidate_path.name}（第 {ckpt.get('epoch', '?')} 代）")
        return model, ckpt
    model, ckpt = ChessNet.from_checkpoint(best_path, device=device)
    print(f"  從 {best_path.name} 開始訓練")
    return model, ckpt


# --- 一個完整的 iteration ---------------------------------------------------


def append_log(path: Path, row: dict) -> None:
    """把一個 iteration 的統計附加到 logs/selfplay_log.csv。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def print_health(stats: dict[str, float]) -> None:
    """印出健康指標，並在超標時直接講該調什麼（規格 §7.6）。"""
    print(f"  盤面數       : {stats['positions']:,}")
    print(f"  平均步數     : {stats['avg_plies']:.1f}")
    print(f"  和局比例     : {stats['draw_rate'] * 100:.1f} %")
    print(f"  認輸比例     : {stats['resign_rate'] * 100:.1f} %")
    print(f"  認輸誤判率   : {stats['false_resign_rate'] * 100:.1f} %（審計局統計）")
    print(f"  截斷機率質量 : {stats['avg_truncated_mass']:.4f}")

    if stats["draw_rate"] > DRAW_RATE_WARN:
        print(
            f"  [警告] 和局比例超過 {DRAW_RATE_WARN * 100:.0f} %，代表模型過度保守。\n"
            f"         調高 config.yaml 的 mcts.dirichlet_epsilon（目前 0.25 → 試 0.35）\n"
            f"         或加大 mcts.temperature_moves。"
        )
    if stats["avg_truncated_mass"] > TRUNCATION_WARN:
        print(
            f"  [警告] 被截斷的機率質量平均 {stats['avg_truncated_mass']:.3f} > {TRUNCATION_WARN}，\n"
            f"         代表 SPARSE_POLICY_K={SPARSE_POLICY_K} 太小，請調高 src/preprocess.py 的常數。"
        )
    if stats["false_resign_rate"] > 0.05:
        print(
            f"  [警告] 認輸誤判率 {stats['false_resign_rate'] * 100:.1f} % > 5 %，\n"
            f"         把 config.yaml 的 selfplay.resign_threshold 調得更嚴（-0.9 → -0.95）。"
        )


def run_iteration(
    iteration: int,
    cfg: Config,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """跑一個完整的 iteration（規格 §7.3 的四個步驟）。"""
    started = time.perf_counter()
    buffer_dir = cfg.resolve_path(args.buffer_dir)
    best_path = PROJECT_ROOT / "models" / "best.pt"
    candidate_path = PROJECT_ROOT / "models" / "selfplay_candidate.pt"

    print(f"\n{'=' * 62}\niteration {iteration}\n{'=' * 62}")

    # 1. 產生對局 —— 一律用 best.pt（已經通過把關的那個），不是正在訓練的那個
    generator, _ = ChessNet.from_checkpoint(best_path, device=device)
    generator.eval()
    shard = buffer_dir / f"iter_{iteration:04d}.npy"

    def flush(partial: list[tuple]) -> None:
        """把中途結果寫進 shard（同一個檔，覆寫）。"""
        save_shard(partial, shard)

    rows, stats = generate_games(
        generator, device, cfg, args.games, args.simulations,
        cfg.seed + iteration, on_progress=flush,
    )
    print_health(stats)

    # 2. 寫入 replay buffer
    save_shard(rows, shard)
    removed = prune_buffer(buffer_dir)
    print(f"  已寫入 {shard}（{len(rows):,} 筆）")
    if removed:
        print(f"  滑動視窗刪掉 {len(removed)} 個舊 shard")

    row = {"iteration": iteration, **stats, "seconds": round(time.perf_counter() - started, 1)}

    if args.no_train:
        append_log(PROJECT_ROOT / "logs" / LOG_FILE_NAME, row)
        return row

    # 3. 訓練 —— 在「持續訓練的那個模型」上繼續，不是每代從 best.pt 重來
    model, ckpt = load_training_model(best_path, candidate_path, device, args.from_best)
    optimizer = make_optimizer(model, cfg, args.lr)
    if ckpt.get("optimizer_state_dict") and not args.from_best:
        # Adam 的動量也要接續，不然每代開頭都會有一段震盪
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    losses = train_on_buffer(
        model, optimizer, device, cfg, buffer_dir,
        steps=args.train_steps, batch_size=args.batch_size,
        seed=cfg.seed + iteration,
    )
    # **checkpoint 裡記的架構必須跟權重一致。**
    # 這裡曾經直接寫 `cfg.to_dict()`，但 cfg 來自 config.yaml 的預設 preset，
    # 跟「實際載進來的那個模型」可能是不同大小的網路：
    # best.pt 是 small(C96)、config.yaml 預設 base(C128)，存下去之後
    # 下一代 `from_checkpoint` 會照 C128 建模型再去載 C96 的權重，
    # 直接 size mismatch 掛掉（實測第 2 代就爆了）。
    # 正確做法是沿用「載進來那個 checkpoint 的 config」。
    saved_config = ckpt.get("config") or cfg.to_dict()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": iteration,
            "global_step": ckpt.get("global_step", 0) + args.train_steps,
            "config": saved_config,
            "source": "selfplay",
        },
        candidate_path,
    )
    print(f"  訓練完成：policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f}")
    print(f"  候選模型存到 {candidate_path}")
    row.update(train_steps=args.train_steps, **losses)

    # 4. 把關：SPRT 對打舊版，通過才更新 best.pt
    if args.keep_best:
        print("  （--keep-best：候選留在 selfplay_candidate.pt，不動 best.pt）")
        row.update(accepted=False, sprt_result="skipped")
    elif args.no_gate:
        print("  [警告] --no-gate：沒有經過 SPRT 就直接覆蓋 best.pt。")
        print("         正式跑迴圈時不要用這個旗標 —— 沒把關的模型會污染後續所有自我對弈資料。")
        torch.save(torch.load(candidate_path, weights_only=False), best_path)
        row.update(accepted=True, sprt_result="skipped")
    else:
        row.update(_run_gate(cfg, candidate_path, best_path, args.gate_rounds))

    row["seconds"] = round(time.perf_counter() - started, 1)
    append_log(PROJECT_ROOT / "logs" / LOG_FILE_NAME, row)
    return row


def _run_gate(cfg: Config, candidate: Path, best: Path, rounds: int) -> dict:
    """SPRT 對打：新模型贏了才取代 best.pt。

    沒通過就保留舊版，用新資料繼續訓練（規格 §7.3 第 4 步）——
    這是防止「訓練損失下降但棋力變差」的唯一防線，不要為了跑快而拿掉。
    """
    from src.evaluate import evaluate_sprt

    print(f"\n  SPRT 把關：候選 vs 目前的 best（最多 {rounds} 輪）")
    try:
        result = evaluate_sprt(
            cfg, str(candidate), str(best), rounds,
            new_name="candidate", old_name="best",
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"  [警告] SPRT 跑不起來（{exc}），這一代保留舊的 best.pt。")
        return {"accepted": False, "sprt_result": f"error: {exc}"}

    # **只有 H1 才算通過。** 沒結論就當作沒變強：寧可多跑一代，
    # 也不要把一個沒被證明比較強的模型放進去污染後續所有自我對弈資料。
    conclusion = result.get("sprt_conclusion", "?")
    accepted = conclusion == "H1"
    elo = float(result.get("elo", 0.0))
    if accepted:
        torch.save(torch.load(candidate, weights_only=False), best)
        print(f"  ✓ 候選勝出（{elo:+.1f} Elo），已更新 best.pt")
    else:
        print(f"  ✗ 候選沒有被證明比較強（{elo:+.1f} Elo，SPRT={conclusion}），保留舊的 best.pt")
    return {"accepted": accepted, "elo": round(elo, 1), "sprt_result": conclusion}


def main() -> None:
    """`python -m src.selfplay --help` 看全部參數。"""
    parser = argparse.ArgumentParser(description="自我對弈訓練迴圈（Phase 2 §7）")
    add_common_args(parser)
    parser.add_argument("--iterations", type=int, default=1, help="跑幾個 iteration")
    parser.add_argument("--games", type=int, default=None, help="每個 iteration 下幾局")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--train-steps", type=int, default=2000, help="每個 iteration 訓練幾步")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--lr", type=float, default=1.0e-4,
        help="自我對弈的學習率要比監督式小很多，否則會把既有棋力洗掉",
    )
    parser.add_argument("--gate-rounds", type=int, default=60, help="SPRT 最多打幾輪")
    parser.add_argument("--buffer-dir", type=str, default=SELFPLAY_DIR, help="replay buffer 目錄")
    parser.add_argument("--no-train", action="store_true", help="只產生對局，不訓練")
    parser.add_argument(
        "--no-gate", action="store_true",
        help="跳過 SPRT 直接接受新模型（危險，只在除錯時用）",
    )
    parser.add_argument(
        "--keep-best", action="store_true",
        help="不論結果都不動 models/best.pt（候選留在 selfplay_candidate.pt）",
    )
    parser.add_argument(
        "--from-best", action="store_true",
        help="從 best.pt 重新開始訓練，不接續 selfplay_candidate.pt",
    )
    parser.add_argument("--smoke-test", action="store_true", help="4 局 + 50 步，確認整條路是通的")
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)

    if args.games is None:
        args.games = int((cfg.selfplay or {}).get("games_per_iteration", 500))
    if args.smoke_test:
        args.games, args.train_steps, args.simulations = 4, 50, 100
        args.batch_size = min(args.batch_size, 64)
        # 用另一個 buffer 目錄，才不會把煙霧測試的爛資料混進正式的 replay buffer
        args.buffer_dir = SELFPLAY_DIR + "_smoke"
        args.keep_best = True
        args.from_best = True      # 煙霧測試不要接續正式的候選模型
        print("[smoke-test] 4 局、100 模擬、50 步訓練、跳過 SPRT、不動 best.pt")

    buffer_dir = cfg.resolve_path(args.buffer_dir)
    print(f"裝置        : {device}")
    print(f"每代局數    : {args.games}（每步 {args.simulations} 次模擬）")
    print(f"訓練        : {args.train_steps} 步 × batch {args.batch_size}, lr={args.lr}")
    print(f"replay buffer: {buffer_dir}（保留 {BUFFER_ITERATIONS} 代）")

    existing = len(buffer_files(buffer_dir))
    for i in range(args.iterations):
        run_iteration(existing + i + 1, cfg, device, args)

    print(f"\n完成。統計在 logs/{LOG_FILE_NAME}")


if __name__ == "__main__":
    main()
