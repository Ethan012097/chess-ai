"""torch Dataset / DataLoader：把 70 bytes 的緊湊表示展開成 (18,8,8) 張量。

GPU 訓練的瓶頸通常在資料供給端，所以 `__getitem__` 有兩條鐵律：
  1. **只用純 NumPy 向量化**，絕對不呼叫 python-chess
  2. 不做任何 I/O（資料透過 np.memmap 讀，由作業系統的 page cache 處理）

Windows 上 DataLoader 的 worker 會 re-import 主模組，所以任何用到這裡的腳本，
進入點都必須包在 `if __name__ == "__main__":` 裡。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config, add_common_args, load_config
from src.encoding import (
    BOARD_SIZE,
    NUM_INPUT_PLANES,
    NUM_MOVES,
    PLANE_EN_PASSANT,
    PLANE_HALFMOVE,
    PLANE_OPP_KINGSIDE,
    PLANE_OPP_QUEENSIDE,
    PLANE_OWN_KINGSIDE,
    PLANE_OWN_QUEENSIDE,
    HALFMOVE_SCALE,
)
from src.preprocess import (
    POSITION_DTYPE,
    SELFPLAY_DTYPE,
    VALUE_QUANT_SCALE,
    decode_sparse_policy,
)

# 易位權的 bit → plane 對照（順序要跟 encode_board_compact 一致）
CASTLING_BIT_TO_PLANE: tuple[int, ...] = (
    PLANE_OWN_KINGSIDE,
    PLANE_OWN_QUEENSIDE,
    PLANE_OPP_KINGSIDE,
    PLANE_OPP_QUEENSIDE,
)


class ChessPositionDataset(Dataset):
    """從 .npy memmap 讀盤面，回傳 (board_tensor, policy_target, value_target)。

    這裡必須是 class，因為 PyTorch 的 DataLoader 就是吃 Dataset 介面
    （要有 `__len__` 與 `__getitem__`）。

    **一個 class 吃兩種檔案格式**（用 dtype 自動判斷，不用多傳參數）：

    | 檔案來源 | dtype | policy target |
    |---|---|---|
    | 監督式（preprocess.py） | `POSITION_DTYPE`（70 bytes） | 人類走的那一步，單一 index |
    | 自我對弈（selfplay.py） | `SELFPLAY_DTYPE`（197 bytes） | MCTS 訪問次數分佈，稀疏存前 32 名 |

    `soft_targets` 決定 target 的形狀：
      - False：純量 int64 的 move index，配 `CrossEntropyLoss`
      - True ：4672 維機率向量

    自我對弈的資料**只能**配 `soft_targets=True`——它本來就沒有「唯一正確的一步」，
    硬要取 argmax 等於把 MCTS 想過的東西全丟掉。
    """

    def __init__(
        self,
        path: str | Path,
        soft_targets: bool = False,
        max_samples: int | None = None,
    ) -> None:
        """
        Args:
            path: train.npy / val.npy / 自我對弈的 shard 路徑。
            soft_targets: 見上方說明。
            max_samples: 只用前 N 筆（--smoke-test 與快速評估用）。
        """
        self.path = Path(path)
        self.soft_targets = soft_targets

        if not self.path.exists():
            raise FileNotFoundError(
                f"找不到資料檔：{self.path}\n"
                f"下一步：\n"
                f"  1. 下載棋譜   python scripts/download_data.py --source elite --month 2024-01\n"
                f'  2. 執行前處理 python -m src.preprocess --input "data/raw/*.pgn"'
            )

        # mmap_mode="r" 回傳的就是 np.memmap，不會把整個檔案讀進記憶體
        self.data = np.load(self.path, mmap_mode="r")
        self.is_selfplay = self.data.dtype == SELFPLAY_DTYPE
        if not self.is_selfplay and self.data.dtype != POSITION_DTYPE:
            raise ValueError(
                f"{self.path} 的 dtype 不對（{self.data.dtype}）。\n"
                f"認得的格式只有兩種：POSITION_DTYPE（監督式）與 SELFPLAY_DTYPE（自我對弈）。\n"
                f"這個檔案可能是舊版格式，請重跑 preprocess.py 重新產生。"
            )
        if self.is_selfplay and not soft_targets:
            raise ValueError(
                f"{self.path} 是自我對弈資料，policy target 是機率分佈，"
                f"必須搭配 soft_targets=true。\n"
                f"請在 config.yaml 把 train.soft_targets 設成 true，"
                f"或改用 python -m src.selfplay --train。"
            )

        self.length = len(self.data) if max_samples is None else min(max_samples, len(self.data))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把第 idx 筆展開成訓練樣本。

        Args:
            idx: 0 <= idx < len(self)。

        Returns:
            (board, policy_target, value_target)
              board:         (18, 8, 8) float32
              policy_target: soft_targets=False 時是純量 int64（move index）；
                             True 時是 (4672,) float32 機率向量
              value_target:  純量 float32，值域 {-1, 0, +1}
        """
        row = self.data[idx]
        planes = self._decode_planes(row)

        if self.is_selfplay:
            # 自我對弈：value 是量化過的 int16，policy 是稀疏的前 32 名
            value_target = torch.tensor(
                float(row["value_target"]) / VALUE_QUANT_SCALE, dtype=torch.float32
            )
            dense = decode_sparse_policy(row["top_indices"], row["top_probs"], NUM_MOVES)
            return torch.from_numpy(planes), torch.from_numpy(dense), value_target

        value_target = torch.tensor(float(row["result"]), dtype=torch.float32)

        if self.soft_targets:
            # 監督式資料只有「人類走的那一步」，做成 one-hot 就是它的機率分佈
            policy_target = torch.zeros(NUM_MOVES, dtype=torch.float32)
            policy_target[int(row["move_index"])] = 1.0
        else:
            policy_target = torch.tensor(int(row["move_index"]), dtype=torch.long)

        return torch.from_numpy(planes), policy_target, value_target

    @staticmethod
    def _decode_planes(row: np.void) -> np.ndarray:
        """把一筆緊湊表示展開成 (18, 8, 8) float32。

        跟 `encoding.decode_compact_to_planes` 的邏輯相同，但直接吃 structured
        array 的一列，少一層轉換。這支每個樣本都會跑一次，所以寫得盡量精簡。

        Args:
            row: POSITION_DTYPE 的一列。

        Returns:
            (18, 8, 8) float32。
        """
        planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # planes 0–11：棋子。code 1..12 對應 plane 0..11
        codes = row["pieces"].astype(np.int64)
        squares = np.flatnonzero(codes)
        if squares.size:
            planes[
                codes[squares] - 1,
                squares >> 3,        # rank = square // 8
                squares & 7,         # file = square % 8
            ] = 1.0

        # planes 12–15：易位權，整層填 1
        castling = int(row["castling"])
        for bit, plane_idx in enumerate(CASTLING_BIT_TO_PLANE):
            if castling & (1 << bit):
                planes[plane_idx] = 1.0

        # plane 16：吃過路兵目標格
        ep = int(row["ep_square"])
        if ep >= 0:
            planes[PLANE_EN_PASSANT, ep >> 3, ep & 7] = 1.0

        # plane 17：五十步計數 / 100
        planes[PLANE_HALFMOVE] = float(row["halfmove"]) / HALFMOVE_SCALE

        return planes


def build_dataloader(
    path: str | Path,
    cfg: Config,
    shuffle: bool,
    max_samples: int | None = None,
    batch_size: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """建立 DataLoader，參數依 config.yaml。

    Args:
        path: train.npy / val.npy。
        cfg: 設定。
        shuffle: train 要 True，val 要 False。
        max_samples: 只用前 N 筆。
        batch_size: 覆寫 cfg.train.batch_size。
        drop_last: 丟掉最後不滿一個 batch 的資料（訓練時建議 True，形狀固定才好用
            cudnn.benchmark）。

    Returns:
        DataLoader。
    """
    dataset = ChessPositionDataset(
        path, soft_targets=cfg.train.soft_targets, max_samples=max_samples
    )
    num_workers = cfg.train.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        # persistent_workers 在 num_workers=0 時會報錯，要一起判斷
        persistent_workers=cfg.train.persistent_workers and num_workers > 0,
        drop_last=drop_last,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def main() -> None:
    """`python -m src.dataset` 印出資料集大小並抽幾筆檢查形狀。"""
    parser = argparse.ArgumentParser(description="檢查前處理產生的資料集")
    add_common_args(parser)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--num", type=int, default=3, help="抽幾筆出來看")
    parser.add_argument("--benchmark", action="store_true", help="量測 __getitem__ 的吞吐量")
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    path = cfg.resolve_path(cfg.data.train_file if args.split == "train" else cfg.data.val_file)

    dataset = ChessPositionDataset(path, soft_targets=cfg.train.soft_targets)
    size_mb = path.stat().st_size / (1 << 20)
    print(f"檔案      : {path}")
    print(f"盤面數    : {len(dataset):,}（{size_mb:.1f} MB）")
    print(f"soft_targets: {cfg.train.soft_targets}")

    print(f"\n抽 {args.num} 筆檢查：")
    for i in range(min(args.num, len(dataset))):
        board, policy, value = dataset[i]
        policy_desc = (
            f"one-hot {tuple(policy.shape)}" if policy.dim() else f"index {int(policy)}"
        )
        print(
            f"  [{i}] board={tuple(board.shape)} {board.dtype}  "
            f"policy={policy_desc}  value={float(value):+.0f}  "
            f"棋子數={int(board[:12].sum())}"
        )

    if args.benchmark:
        import time

        n = min(20000, len(dataset))
        start = time.perf_counter()
        for i in range(n):
            dataset[i]
        elapsed = time.perf_counter() - start
        print(f"\n單執行緒 __getitem__：{n / elapsed:,.0f} 盤面/秒")
        print(f"（num_workers={cfg.train.num_workers} 時大約要再乘上 worker 數）")


if __name__ == "__main__":
    main()
