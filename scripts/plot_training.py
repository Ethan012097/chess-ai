"""畫訓練曲線（讀 logs/train_log.csv）。

用法：
    python scripts/plot_training.py
    python scripts/plot_training.py --out logs/curves.png

刻意不引入 wandb / tensorboard：csv + matplotlib 就夠了，而且不用連網、不用登入。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不需要 GUI，直接存檔
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "logs" / "train_log.csv"
DEFAULT_VAL_LOG = PROJECT_ROOT / "logs" / "val_log.csv"
DEFAULT_OUT = PROJECT_ROOT / "logs" / "training_curves.png"


def read_log(path: Path) -> dict[str, list[float]]:
    """讀 csv，回傳 {欄位名: 數值 list}。空值會被跳過。"""
    columns: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value in ("", None):
                    continue
                try:
                    columns.setdefault(key, []).append(float(value))
                except ValueError:
                    pass
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description="把 train_log.csv 畫成訓練曲線")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG), help="訓練 csv 路徑")
    parser.add_argument(
        "--val-log", type=str, default=str(DEFAULT_VAL_LOG), help="驗證 csv 路徑（有才畫）"
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="輸出圖檔路徑")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(
            f"找不到 {log_path}\n"
            f"下一步：先訓練一下產生紀錄\n"
            f"  python -m src.train --smoke-test"
        )

    data = read_log(log_path)
    if not data.get("step"):
        raise SystemExit(f"{log_path} 沒有資料。先跑 python -m src.train --smoke-test")

    steps = data["step"]

    # val_log.csv 是選配的：有就一起畫上去，方便看 train / val 有沒有拉開（過擬合）
    val_path = Path(args.val_log)
    val = read_log(val_path) if val_path.exists() else {}
    val_steps = val.get("step", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Chess AI training curves", fontsize=14)

    # 左上：兩個 loss
    ax = axes[0][0]
    if "policy_loss" in data:
        ax.plot(steps, data["policy_loss"], label="policy loss", linewidth=1)
    if "value_loss" in data:
        ax.plot(steps, data["value_loss"], label="value loss", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右上：policy 準確率
    ax = axes[0][1]
    if "policy_top1" in data:
        ax.plot(steps, [v * 100 for v in data["policy_top1"]], label="top-1 (train)", linewidth=1)
    if "policy_top5" in data:
        ax.plot(steps, [v * 100 for v in data["policy_top5"]], label="top-5 (train)", linewidth=1)
    if val_steps and "policy_top1" in val:
        ax.plot(
            val_steps, [v * 100 for v in val["policy_top1"]],
            "o--", label="top-1 (val)", linewidth=1, markersize=4,
        )
    if val_steps and "policy_top5" in val:
        ax.plot(
            val_steps, [v * 100 for v in val["policy_top5"]],
            "o--", label="top-5 (val)", linewidth=1, markersize=4,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Policy accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    # 左下：value MAE
    ax = axes[1][0]
    if "value_mae" in data:
        ax.plot(steps, data["value_mae"], color="tab:green", linewidth=1, label="train")
    if val_steps and "value_mae" in val:
        ax.plot(
            val_steps, val["value_mae"], "o--",
            color="tab:olive", linewidth=1, markersize=4, label="val",
        )
        ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("MAE")
    ax.set_title("Value MAE")
    ax.grid(alpha=0.3)

    # 右下：學習率
    ax = axes[1][1]
    if "lr" in data:
        ax.plot(steps, data["lr"], color="tab:red", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.set_title("Learning rate")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"已存圖：{out_path}")

    # 順便印出最後一筆，方便快速看有沒有訓歪
    if "policy_top1" in data:
        print(
            f"最後一筆：step={steps[-1]:.0f} "
            f"top1={data['policy_top1'][-1] * 100:.2f}% "
            f"top5={data.get('policy_top5', [0])[-1] * 100:.2f}% "
            f"value_mae={data.get('value_mae', [0])[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
