"""PGN → 緊湊二進位檔（.npy memmap）。

用法（在專案根目錄執行）：
    python -m src.preprocess --input "data/raw/*.pgn"
    python -m src.preprocess --input "data/raw/*.pgn" --max-positions 200000
    python -m src.preprocess --input a.pgn b.pgn --max-positions 30000000

為什麼不直接存 (18,8,8) float32：那是 4.6 KB/盤面，1500 萬盤面要 69 GB，硬碟會爆。
這裡每個盤面只存 70 bytes（1500 萬盤面約 1 GB），到 `Dataset.__getitem__` 才用純
NumPy 展開成張量。

中斷續跑：處理過程分 shard 寫到 data/processed/shards/，並用 manifest.json 記錄
已完成的輸入檔。重跑同一個指令會跳過已完成的檔案，只補沒做完的部分。
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from src.config import Config, add_common_args, load_config
from src.encoding import encode_board_compact, encode_move

# 每個盤面 70 bytes。numpy 的 structured dtype 預設不做對齊（packed），
# 所以 itemsize 剛好是 64 + 1 + 1 + 1 + 2 + 1 = 70。
POSITION_DTYPE = np.dtype(
    [
        ("pieces", np.int8, (64,)),   # 0=空, 1..6=己方 PNBRQK, 7..12=對方 PNBRQK（已鏡射）
        ("castling", np.uint8),       # 4 個 bit
        ("ep_square", np.int8),       # -1 代表沒有
        ("halfmove", np.uint8),       # 五十步計數
        ("move_index", np.uint16),    # 0..4671，人類實際走的那步
        ("result", np.int8),          # +1 己方贏, 0 和, -1 己方輸（當前走棋方視角）
    ]
)

# --- 自我對弈的儲存格式（Phase 2 §7.2）-------------------------------------
# 放在這裡而不是 selfplay.py：所有「磁碟上長什麼樣」的定義集中在同一個檔比較好找，
# 而且 dataset.py 只要從這裡 import 就好，不會跟 selfplay.py 互相 import。
#
# 為什麼要稀疏：MCTS 的 policy target 是 4672 維機率分佈，存成 float32 是
# 18 KB/盤面，100 萬盤面就 18 GB，完全不可行。但訪問次數本來就極度集中在少數
# 幾個著法上，取前 K 名就夠了。合法著法數通常 30–40，K=32 幾乎不會丟掉有意義的
# 機率質量（前處理時會統計被截斷掉的量，見 `encode_sparse_policy`）。
SPARSE_POLICY_K = 32
# 機率用 uint16 存：× 65535 後取整，解析度 1.5e-5，遠低於訓練會在意的精度
PROB_QUANT_SCALE = 65535
# value 用 int16 存：× 10000，值域 -1 ~ +1 → -10000 ~ +10000
VALUE_QUANT_SCALE = 10000

# 每個盤面 197 bytes（64 + 1 + 1 + 1 + 64 + 64 + 2）。
SELFPLAY_DTYPE = np.dtype(
    [
        ("pieces", np.int8, (64,)),                      # 同 Phase 1
        ("castling", np.uint8),
        ("ep_square", np.int8),
        ("halfmove", np.uint8),
        ("top_indices", np.uint16, (SPARSE_POLICY_K,)),  # 訪問次數前 32 名的著法 index
        ("top_probs", np.uint16, (SPARSE_POLICY_K,)),    # 機率 × 65535 後取整
        ("value_target", np.int16),                      # 值 × 10000（當前走棋方視角）
    ]
)

SHARD_DIR_NAME = "shards"
MANIFEST_NAME = "manifest.json"
STATS_EVERY_GAMES = 10_000          # 每處理這麼多局印一次統計
PROGRESS_UPDATE_GAMES = 100         # tqdm 更新頻率
MAX_CONSECUTIVE_PARSE_ERRORS = 100  # 連續這麼多局解析失敗就放棄整個檔案

# PGN 的 Result 標籤 → 白方視角的分數
RESULT_TO_WHITE_SCORE: dict[str, int] = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

# 被排除的 Termination 值（小寫比對）
BAD_TERMINATIONS = ("abandoned", "rules infraction")


# --- 稀疏 policy 的編碼 / 解碼（Phase 2 §7.2）-------------------------------


def encode_sparse_policy(
    probs: dict[int, float], k: int = SPARSE_POLICY_K
) -> tuple[np.ndarray, np.ndarray, float]:
    """把 4672 維的機率分佈壓成「前 k 名的 index + 量化機率」。

    Args:
        probs: {著法 index: 機率}。通常是 MCTS 的訪問次數正規化後的結果。
            不需要事先排序，也不要求總和剛好是 1。
        k: 保留前幾名。

    Returns:
        (top_indices, top_probs, truncated_mass)
          top_indices: (k,) uint16，不足的補 0
          top_probs:   (k,) uint16，機率 × 65535 取整，不足的補 0
          truncated_mass: **被丟掉的機率總和**。平均超過 0.02 就代表 k 太小
            （規格 §7.2），`selfplay.py` 每個 iteration 都會把這個數字印出來。
    """
    top_indices = np.zeros(k, dtype=np.uint16)
    top_probs = np.zeros(k, dtype=np.uint16)
    if not probs:
        return top_indices, top_probs, 0.0

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    kept = ranked[:k]
    total = sum(p for _, p in ranked)
    truncated = sum(p for _, p in ranked[k:])

    for slot, (index, prob) in enumerate(kept):
        top_indices[slot] = index
        top_probs[slot] = min(PROB_QUANT_SCALE, max(0, round(prob * PROB_QUANT_SCALE)))

    # 用比例回報，這樣不管 probs 有沒有正規化過都講得通
    return top_indices, top_probs, float(truncated / total) if total > 0 else 0.0


def decode_sparse_policy(
    top_indices: np.ndarray, top_probs: np.ndarray, num_moves: int
) -> np.ndarray:
    """把稀疏格式展開回 4672 維機率向量。

    未列入的位置補 0，**並重新正規化**（因為被截斷的部分不見了，剩下的加起來
    不會剛好是 1；不正規化的話損失的尺度會隨盤面浮動）。

    Args:
        top_indices: (k,) uint16。
        top_probs: (k,) uint16。
        num_moves: 輸出維度（4672）。

    Returns:
        (num_moves,) float32，總和為 1（全空時回傳全 0）。
    """
    dense = np.zeros(num_moves, dtype=np.float32)
    mask = top_probs > 0
    if not mask.any():
        return dense
    # 同一個 index 不會重複出現，直接指派即可
    dense[top_indices[mask].astype(np.int64)] = top_probs[mask].astype(np.float32)
    dense /= dense.sum()
    return dense


@dataclass
class FilterStats:
    """統計每個過濾原因擋掉幾局，方便判斷篩選規則是不是設得太嚴。"""

    counter: Counter[str]

    @classmethod
    def new(cls) -> "FilterStats":
        return cls(counter=Counter())

    def hit(self, reason: str) -> None:
        self.counter[reason] += 1

    def report(self) -> str:
        if not self.counter:
            return "（沒有棋局被過濾）"
        total = sum(self.counter.values())
        parts = [f"{k}={v}" for k, v in self.counter.most_common()]
        return f"總共過濾 {total} 局： " + ", ".join(parts)


def parse_time_control_seconds(time_control: str | None) -> int | None:
    """從 PGN 的 TimeControl 取出起始秒數。

    Args:
        time_control: 例如 "600+5"、"180"、"-"、None。

    Returns:
        起始秒數；無法判斷時回傳 None（呼叫端自行決定要不要放行）。
    """
    if not time_control or time_control == "-":
        return None
    base = time_control.split("+")[0].strip()
    try:
        return int(base)
    except ValueError:
        return None


def parse_elo(value: str | None) -> int | None:
    """把 WhiteElo / BlackElo 字串轉成整數，無法解析回傳 None。"""
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def should_keep_game(
    headers: chess.pgn.Headers,
    num_plies: int,
    cfg: Config,
    stats: FilterStats,
) -> bool:
    """套用棋局篩選規則。

    Args:
        headers: PGN 標頭。
        num_plies: 這盤棋的總步數（ply，一方走一步算一步）。
        cfg: 設定。
        stats: 統計器，會記錄被擋掉的原因。

    Returns:
        True 表示保留這盤棋。
    """
    d = cfg.data

    # 變體：Variant 存在且不是 Standard 就排除
    variant = headers.get("Variant")
    if variant and variant.strip().lower() not in ("standard", "chess", ""):
        stats.hit(f"變體({variant})")
        return False

    # Termination
    termination = (headers.get("Termination") or "").strip().lower()
    if termination in BAD_TERMINATIONS:
        stats.hit(f"Termination({termination})")
        return False

    # 結果必須是三種之一（"*" 代表未完成）
    if headers.get("Result") not in RESULT_TO_WHITE_SCORE:
        stats.hit("結果未定(*)")
        return False

    # Elo：兩方都要 ≥ min_elo（elite 來源已滿足，仍再檢查一次）
    white_elo = parse_elo(headers.get("WhiteElo"))
    black_elo = parse_elo(headers.get("BlackElo"))
    if white_elo is None or black_elo is None:
        stats.hit("缺少 Elo")
        return False
    if white_elo < d.min_elo or black_elo < d.min_elo:
        stats.hit(f"Elo<{d.min_elo}")
        return False

    # 排除 bullet
    tc_seconds = parse_time_control_seconds(headers.get("TimeControl"))
    if tc_seconds is not None and tc_seconds < d.min_time_control_seconds:
        stats.hit("bullet")
        return False

    # 步數
    if num_plies < d.min_game_plies:
        stats.hit(f"步數<{d.min_game_plies}")
        return False
    if num_plies > d.max_game_plies:
        stats.hit(f"步數>{d.max_game_plies}")
        return False

    return True


def sample_ply_indices(num_plies: int, cfg: Config, rng: random.Random) -> list[int]:
    """決定這盤棋要取哪幾個 ply 的盤面。

    規則：跳過前 8 步與最後 2 步；每盤最多取 40 個，超過就均勻隨機抽。

    Args:
        num_plies: 這盤棋的總步數。
        cfg: 設定。
        rng: 亂數產生器（傳進來才能用 seed 重現）。

    Returns:
        遞增排序的 ply index 清單（0-based，代表「第幾步之前的盤面」）。
    """
    d = cfg.data
    start = d.skip_opening_plies
    end = num_plies - d.skip_ending_plies  # 不含
    if end <= start:
        return []

    candidates = range(start, end)
    if len(candidates) <= d.max_positions_per_game:
        return list(candidates)
    return sorted(rng.sample(list(candidates), d.max_positions_per_game))


def extract_positions(
    game: chess.pgn.Game,
    cfg: Config,
    rng: random.Random,
    stats: FilterStats,
) -> list[tuple]:
    """把一盤棋轉成一串 POSITION_DTYPE 的 tuple。

    Args:
        game: 已解析的棋局。
        cfg: 設定。
        rng: 亂數產生器。
        stats: 統計器。

    Returns:
        每個元素是 (pieces, castling, ep_square, halfmove, move_index, result)，
        可以直接塞進 POSITION_DTYPE 的陣列。
    """
    moves = list(game.mainline_moves())
    num_plies = len(moves)

    if not should_keep_game(game.headers, num_plies, cfg, stats):
        return []

    white_score = RESULT_TO_WHITE_SCORE[game.headers["Result"]]
    wanted = set(sample_ply_indices(num_plies, cfg, rng))
    if not wanted:
        stats.hit("沒有可取樣的盤面")
        return []

    rows: list[tuple] = []
    board = game.board()
    for ply, move in enumerate(moves):
        if ply in wanted:
            # result 是「當前走棋方」的視角：白方走棋時 = white_score，黑方走棋時取負號
            result = white_score if board.turn == chess.WHITE else -white_score
            try:
                move_index = encode_move(move, board.turn)
            except ValueError:
                # 理論上不該發生（合法著法都編得動），保險起見跳過這個盤面
                stats.hit("著法無法編碼")
                board.push(move)
                continue
            pieces, castling, ep_square, halfmove = encode_board_compact(board)
            rows.append((pieces, castling, ep_square, halfmove, move_index, result))
        board.push(move)

    return rows


def iter_games(pgn_file: TextIO) -> Iterator[chess.pgn.Game]:
    """一局一局讀 PGN。遇到解析錯誤就跳過，不要讓一局壞資料弄死整個前處理。

    連續錯誤達 MAX_CONSECUTIVE_PARSE_ERRORS 次就放棄這個檔案：如果 read_game 一直
    丟例外又沒有往前讀，無限迴圈會讓前處理看起來像卡死。
    """
    consecutive_errors = 0
    while True:
        try:
            game = chess.pgn.read_game(pgn_file)
        except (ValueError, RuntimeError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_PARSE_ERRORS:
                print(
                    f"\n[警告] 連續 {consecutive_errors} 次解析失敗（{exc}），"
                    f"放棄這個檔案剩下的部分。"
                )
                return
            continue
        consecutive_errors = 0
        if game is None:
            return
        yield game


def expand_inputs(patterns: list[str]) -> list[Path]:
    """把命令列給的路徑 / 萬用字元展開成實際檔案清單。

    Windows 的 PowerShell 不會自動展開 *，所以要在 Python 這邊做。
    """
    files: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            files.extend(Path(m) for m in matched)
        elif Path(pattern).exists():
            files.append(Path(pattern))
        else:
            print(f"[警告] 找不到符合的檔案：{pattern}")
    # 去重但保持順序
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


class ShardWriter:
    """把盤面分批寫成 shard 檔案。

    這裡開一個 class 的理由：要同時維護「記憶體緩衝區」與「已寫出的 shard 編號」
    兩個狀態，而且 train / val 各需要一份。用函式傳一堆可變參數反而更難讀。
    """

    def __init__(self, out_dir: Path, split: str, shard_size: int) -> None:
        """
        Args:
            out_dir: shard 存放資料夾。
            split: "train" 或 "val"，用在檔名。
            shard_size: 每個 shard 幾個盤面。
        """
        self.out_dir = out_dir
        self.split = split
        self.shard_size = shard_size
        self.buffer: list[tuple] = []
        self.total_written = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # 續跑時接續已存在的 shard 編號
        existing = sorted(self.out_dir.glob(f"{split}_*.npy"))
        self.shard_index = len(existing)
        for path in existing:
            self.total_written += int(np.load(path, mmap_mode="r").shape[0])

    def add(self, rows: list[tuple]) -> None:
        """加入一批盤面，滿了就自動落盤。"""
        self.buffer.extend(rows)
        while len(self.buffer) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, count: int) -> None:
        chunk = self.buffer[:count]
        self.buffer = self.buffer[count:]
        arr = np.array(chunk, dtype=POSITION_DTYPE)
        path = self.out_dir / f"{self.split}_{self.shard_index:05d}.npy"
        np.save(path, arr)
        self.shard_index += 1
        self.total_written += len(arr)

    def close(self) -> None:
        """把剩下不滿一個 shard 的資料也寫出去。"""
        if self.buffer:
            self._flush(len(self.buffer))

    @property
    def pending(self) -> int:
        return len(self.buffer)


def merge_shards(shard_dir: Path, split: str, dest: Path) -> int:
    """把所有 shard 合併成一個 .npy。

    用 `np.lib.format.open_memmap` 產生「真正的 .npy 檔」，這樣兩種讀法都成立：
      - `np.load(path, mmap_mode="r")`（訓練時用這個，回傳的就是 memmap）
      - 一般的 `np.load(path)`
    如果直接用 `np.memmap` 讀 np.save 出來的檔案，會把 128 bytes 的檔頭當成資料，
    這是很常見的坑，所以這裡統一用 open_memmap。

    Args:
        shard_dir: shard 資料夾。
        split: "train" 或 "val"。
        dest: 輸出的 .npy 路徑。

    Returns:
        合併後的總盤面數。
    """
    shards = sorted(shard_dir.glob(f"{split}_*.npy"))
    if not shards:
        return 0

    total = sum(int(np.load(p, mmap_mode="r").shape[0]) for p in shards)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        dest, mode="w+", dtype=POSITION_DTYPE, shape=(total,)
    )

    offset = 0
    for path in tqdm(shards, desc=f"合併 {split} shards", unit="shard"):
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]
        out[offset : offset + n] = arr
        offset += n
    out.flush()
    del out
    return total


def load_manifest(path: Path) -> dict:
    """讀取續跑用的 manifest。"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done_files": [], "games_kept": 0, "positions": 0}


def save_manifest(path: Path, manifest: dict) -> None:
    """寫回 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def preprocess(
    input_files: list[Path],
    cfg: Config,
    max_positions: int | None = None,
    max_games: int | None = None,
    resume: bool = True,
    keep_shards: bool = False,
) -> tuple[int, int]:
    """主流程：讀 PGN → 篩選 → 取樣 → 寫 shard → 合併成 train.npy / val.npy。

    切分規則：**依棋局**切 train / val（同一盤棋的盤面不可跨集合，否則會洩漏）。

    Args:
        input_files: PGN 檔案清單。
        cfg: 設定。
        max_positions: train 盤面數上限，None 表示不限。
        max_games: 最多處理幾局，None 表示不限（除錯用）。
        resume: True 表示跳過 manifest 裡已完成的檔案。
        keep_shards: True 表示合併後保留 shard 檔案。

    Returns:
        (train 盤面數, val 盤面數)。

    Warning:
        同一個 output_dir 一次只能跑一個 preprocess。合併階段會把 shard 資料夾裡
        **所有** shard 併進去然後刪掉，兩個行程同時寫同一個資料夾會互相吃掉對方的
        資料。要平行跑或做小規模測試，請用 --output-dir 指定不同的資料夾。
    """
    processed_dir = cfg.resolve_path(cfg.data.processed_dir)
    shard_dir = processed_dir / SHARD_DIR_NAME
    manifest_path = shard_dir / MANIFEST_NAME

    manifest = load_manifest(manifest_path) if resume else {
        "done_files": [], "games_kept": 0, "positions": 0
    }
    done_files = set(manifest["done_files"])

    train_writer = ShardWriter(shard_dir, "train", cfg.data.shard_size)
    # val 只佔 val_ratio（預設 2%），如果 shard 大小跟 train 一樣，val 要累積到
    # 1000 萬個 train 盤面才會落盤一次 —— 中途被中斷的話 val 資料就全沒了。
    # 用小很多的 shard 讓 val 也能定期寫到硬碟。
    val_shard_size = max(cfg.data.shard_size // 20, 10_000)
    val_writer = ShardWriter(shard_dir, "val", val_shard_size)

    rng = random.Random(cfg.seed)
    stats = FilterStats.new()
    games_read = 0
    games_kept = manifest["games_kept"]
    positions_total = train_writer.total_written + train_writer.pending

    print(f"輸入檔案 {len(input_files)} 個，已完成 {len(done_files)} 個")
    print(f"目前已有 train 盤面 {train_writer.total_written:,}、val 盤面 {val_writer.total_written:,}")
    if max_positions:
        print(f"train 盤面上限：{max_positions:,}")

    stop = False
    for pgn_path in input_files:
        key = str(pgn_path.resolve())
        if resume and key in done_files:
            print(f"[skip] {pgn_path.name} 已處理過。")
            continue

        print(f"\n[處理] {pgn_path}")
        with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
            bar = tqdm(desc=pgn_path.name, unit="局")
            for game in iter_games(f):
                games_read += 1
                if games_read % PROGRESS_UPDATE_GAMES == 0:
                    bar.update(PROGRESS_UPDATE_GAMES)
                    bar.set_postfix(盤面=f"{positions_total:,}", 保留局數=games_kept)

                rows = extract_positions(game, cfg, rng, stats)
                if rows:
                    games_kept += 1
                    # 依棋局切 train / val
                    if rng.random() < cfg.data.val_ratio:
                        val_writer.add(rows)
                    else:
                        train_writer.add(rows)
                        positions_total += len(rows)

                if games_read % STATS_EVERY_GAMES == 0:
                    tqdm.write(
                        f"  已讀 {games_read:,} 局 / 保留 {games_kept:,} 局 / "
                        f"train 盤面 {positions_total:,}\n  {stats.report()}"
                    )

                if max_positions is not None and positions_total >= max_positions:
                    tqdm.write(f"\n已達盤面上限 {max_positions:,}，停止。")
                    stop = True
                    break
                if max_games is not None and games_read >= max_games:
                    tqdm.write(f"\n已達棋局上限 {max_games:,}，停止。")
                    stop = True
                    break
            bar.close()

        if not stop:
            done_files.add(key)
            manifest["done_files"] = sorted(done_files)
            manifest["games_kept"] = games_kept
            save_manifest(manifest_path, manifest)

        if stop:
            break

    train_writer.close()
    val_writer.close()
    manifest["games_kept"] = games_kept
    manifest["positions"] = train_writer.total_written
    save_manifest(manifest_path, manifest)

    print(f"\n讀取棋局 {games_read:,}，保留 {games_kept:,}")
    print(stats.report())

    print("\n合併 shards…")
    train_file = cfg.resolve_path(cfg.data.train_file)
    val_file = cfg.resolve_path(cfg.data.val_file)
    n_train = merge_shards(shard_dir, "train", train_file)
    n_val = merge_shards(shard_dir, "val", val_file)

    if not keep_shards:
        for p in shard_dir.glob("*.npy"):
            p.unlink()
        print(f"已刪除 shard 檔案（要保留請加 --keep-shards）")

    print(f"\n完成：")
    print(f"  {train_file}  {n_train:,} 盤面（{n_train * POSITION_DTYPE.itemsize / (1 << 20):.1f} MB）")
    print(f"  {val_file}  {n_val:,} 盤面（{n_val * POSITION_DTYPE.itemsize / (1 << 20):.1f} MB）")
    return n_train, n_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 PGN 轉成訓練用的緊湊二進位檔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help='PGN 檔案或萬用字元，可給多個，例如 --input "data/raw/*.pgn"',
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="train 盤面數上限（目標規模 1500 萬–3000 萬）",
    )
    parser.add_argument("--max-games", type=int, default=None, help="最多處理幾局（除錯用）")
    parser.add_argument("--no-resume", action="store_true", help="不要跳過已處理的檔案，全部重來")
    parser.add_argument("--keep-shards", action="store_true", help="合併後保留 shard 檔案")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="輸出資料夾（預設 data/processed）。做小規模測試時請指定另一個資料夾，"
        "否則會跟正在跑的完整前處理搶同一批 shard",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        cfg.data.processed_dir = str(out_dir)
        cfg.data.train_file = str(out_dir / "train.npy")
        cfg.data.val_file = str(out_dir / "val.npy")
    input_files = expand_inputs(args.input)
    if not input_files:
        raise SystemExit(
            "找不到任何 PGN 檔案。\n"
            "下一步：先下載資料\n"
            '  python scripts/download_data.py --source elite --month 2024-01 --sample'
        )

    n_train, n_val = preprocess(
        input_files,
        cfg,
        max_positions=args.max_positions,
        max_games=args.max_games,
        resume=not args.no_resume,
        keep_shards=args.keep_shards,
    )

    if n_train == 0:
        raise SystemExit(
            "\n沒有產生任何盤面。可能原因：\n"
            "  1. PGN 缺少 WhiteElo / BlackElo 標籤（大師棋譜常見）→ 把 config.yaml 的 "
            "data.min_elo 調成 0\n"
            "  2. 棋局太短（< 20 步）\n"
            "上面的過濾統計會告訴你是哪一項擋掉的。"
        )

    print("\n下一步，先確認整條路是通的：")
    print("  python -m src.train --smoke-test")


if __name__ == "__main__":
    main()
