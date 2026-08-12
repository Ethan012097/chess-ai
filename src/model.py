"""雙頭神經網路：policy head（預測人類會走哪一步）+ value head（預測誰會贏）。

結構（AlphaZero 的簡化版，Phase 2 可以直接沿用不用重寫）：

    輸入 (B, 18, 8, 8)
      ↓ Conv2d(18 → C, 3x3, padding=1) + BatchNorm + ReLU
      ↓ ResidualBlock × N
      ├─ Policy head: Conv2d(C → 73, 3x3, padding=1) → flatten → (B, 4672) logits
      └─ Value head : Conv2d(C → 8, 1x1) + BN + ReLU → flatten(512)
                      → Linear(512→256) + ReLU → Linear(256→1) → tanh → (B, 1)

C = channels、N = blocks，由 config.yaml 的 preset 決定（預設 base：C=128、N=10）。

policy head 輸出的是 **raw logits**，softmax 與 legal mask 都在外面做
（訓練時 CrossEntropyLoss 自己會做 softmax，推論時要先套 legal mask 再 softmax）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config, add_common_args, load_config
from src.encoding import NUM_INPUT_PLANES, NUM_MOVE_PLANES, NUM_MOVES

BOARD_SIZE = 8
CONV_KERNEL_SIZE = 3
CONV_PADDING = 1


class ResidualBlock(nn.Module):
    """一個標準的殘差塊：Conv-BN-ReLU-Conv-BN-(+輸入)-ReLU。

    這裡要開 class 是因為 PyTorch 的模組系統要求：有可訓練參數的東西必須是
    nn.Module，才會被 `model.parameters()` 找到、才能存進 state_dict。
    """

    def __init__(self, channels: int) -> None:
        """
        Args:
            channels: 輸入與輸出的通道數（殘差相加要求兩者相同）。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, CONV_KERNEL_SIZE, padding=CONV_PADDING, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, CONV_KERNEL_SIZE, padding=CONV_PADDING, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, 8, 8)

        Returns:
            (B, C, 8, 8)，與輸入同形狀。
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual          # 殘差相加
        return F.relu(out)


class ChessNet(nn.Module):
    """policy + value 雙頭網路。

    這是整個專案唯一的模型類別。Phase 1 用 policy head 直接下棋，
    Phase 2 的 MCTS 會同時用到 policy（當作 prior）與 value（當作葉節點評估）。
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 10,
        value_head_channels: int = 8,
        value_hidden: int = 256,
    ) -> None:
        """
        Args:
            channels: 主幹的通道數 C。
            blocks: 殘差塊數量 N。
            value_head_channels: value head 的 1x1 conv 輸出通道數。
            value_hidden: value head 隱藏層寬度。
        """
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        # --- 主幹 ---
        self.stem_conv = nn.Conv2d(
            NUM_INPUT_PLANES, channels, CONV_KERNEL_SIZE, padding=CONV_PADDING, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(channels)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(blocks)]
        )

        # --- Policy head ---
        # 直接輸出 73 個 plane，flatten 後就是 4672 維。
        # 注意 flatten 的順序必須跟 encoding.py 的 index 定義一致：
        #   index = from_square * 73 + plane，而 from_square = rank * 8 + file，
        # 所以要把 (B, 73, 8, 8) 轉成 (B, 8, 8, 73) 再 flatten。
        self.policy_conv = nn.Conv2d(
            channels, NUM_MOVE_PLANES, CONV_KERNEL_SIZE, padding=CONV_PADDING
        )

        # --- Value head ---
        self.value_conv = nn.Conv2d(channels, value_head_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_head_channels)
        value_flat = value_head_channels * BOARD_SIZE * BOARD_SIZE
        self.value_fc1 = nn.Linear(value_flat, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向傳播。

        Args:
            x: (B, 18, 8, 8) float32，`encode_board` 的輸出。

        Returns:
            (policy_logits, value)
              policy_logits: (B, 4672) raw logits，**沒有** softmax、**沒有** legal mask
              value: (B, 1)，經過 tanh，值域 [-1, 1]，+1 代表「當前走棋方會贏」
        """
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.residual_blocks:
            out = block(out)

        # policy：(B, 73, 8, 8) → (B, 8, 8, 73) → (B, 4672)
        policy = self.policy_conv(out)
        policy_logits = policy.permute(0, 2, 3, 1).reshape(-1, NUM_MOVES)

        # value：(B, 8, 8, 8) → flatten → 256 → 1 → tanh
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.flatten(start_dim=1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def count_parameters(self) -> int:
        """可訓練參數量，訓練開始時會印出來。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: Config) -> "ChessNet":
        """依 config.yaml（已套用 preset）建立模型。"""
        return cls(
            channels=cfg.model.channels,
            blocks=cfg.model.blocks,
            value_head_channels=cfg.model.value_head_channels,
            value_hidden=cfg.model.value_hidden,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: torch.device | str = "cpu"
    ) -> tuple["ChessNet", dict[str, Any]]:
        """從 checkpoint 載入模型。

        Args:
            path: models/best.pt 之類的路徑。
            device: 要載到哪個裝置（不要寫死 cuda，測試要能在 CPU 跑）。

        Returns:
            (model, checkpoint)。checkpoint 裡有 epoch / global_step / config，
            續訓時會用到。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 checkpoint：{path}\n"
                f"下一步：先訓練模型\n"
                f"  python -m src.train --smoke-test    # 先確認路是通的\n"
                f"  python -m src.train                 # 完整訓練"
            )
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_cfg = checkpoint.get("config", {}).get("model", {})
        model = cls(
            channels=model_cfg.get("channels", 128),
            blocks=model_cfg.get("blocks", 10),
            value_head_channels=model_cfg.get("value_head_channels", 8),
            value_hidden=model_cfg.get("value_hidden", 256),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint


def resolve_device(device_setting: str) -> torch.device:
    """把 config 的 device 設定轉成 torch.device。

    Args:
        device_setting: "auto" / "cuda" / "cpu"。

    Returns:
        torch.device。"auto" 會在有 CUDA 時選 cuda，否則 cpu。
    """
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_setting == "cuda" and not torch.cuda.is_available():
        print(
            "[警告] config 指定 cuda 但 torch.cuda.is_available() 是 False，改用 CPU。\n"
            "       多半是裝到 CPU 版 wheel，請先 pip uninstall torch，再照 README 用\n"
            "       --index-url https://download.pytorch.org/whl/cu126 重裝。"
        )
        return torch.device("cpu")
    return torch.device(device_setting)


def main() -> None:
    """`python -m src.model` 印出模型結構與參數量，順便確認 forward 形狀正確。"""
    parser = argparse.ArgumentParser(description="印出模型結構與參數量")
    add_common_args(parser)
    parser.add_argument("--summary", action="store_true", help="連同每層結構一起印出")
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)

    model = ChessNet.from_config(cfg).to(device)
    print(f"preset      : {cfg.preset}")
    print(f"channels    : {model.channels}")
    print(f"blocks      : {model.blocks}")
    print(f"參數量      : {model.count_parameters():,}")
    print(f"裝置        : {device}")

    if args.summary:
        print(model)

    dummy = torch.zeros(2, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, device=device)
    model.eval()
    with torch.no_grad():
        policy_logits, value = model(dummy)
    print(f"\n前向傳播測試：")
    print(f"  輸入          {tuple(dummy.shape)}")
    print(f"  policy_logits {tuple(policy_logits.shape)}")
    print(f"  value         {tuple(value.shape)}，值域 [{value.min():.3f}, {value.max():.3f}]")


if __name__ == "__main__":
    main()
