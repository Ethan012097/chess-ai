"""把 logs/selfplay_log.csv 畫成監控圖（規格 §7.6）。

    python scripts/plot_selfplay.py

**和局比例是最重要的健康指標。** 超過 70 % 代表模型過度保守
（常見於 value 主導、policy 多樣性不足），要調高 mcts.dirichlet_epsilon
或 mcts.temperature_moves。圖上會畫一條紅色的 70 % 參考線。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

# 沒有視窗環境也要能存檔，所以在 import pyplot 之前先切成 Agg
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "logs" / "selfplay_log.csv"
DEFAULT_OUT = PROJECT_ROOT / "logs" / "selfplay_curves.png"

# 規格 §7.6 的警戒線
DRAW_RATE_WARN = 0.70
FALSE_RESIGN_WARN = 0.05


def read_log(path: Path) -> dict[str, list[float]]:
    """讀 csv，回傳 {欄位名: 數值 list}。空值與非數字會被跳過。

    （跟 plot_training.py 的同名函式一樣。刻意各留一份：scripts/ 不是套件，
    互相 import 要動 sys.path，為了 12 行程式不值得。）
    """
    columns: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if value in ("", None):
                    continue
                try:
                    columns.setdefault(key, []).append(float(value))
                except ValueError:
                    pass
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description="把 selfplay_log.csv 畫成監控圖")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(
            f"找不到 {path}\n"
            f"下一步：先跑一個 iteration 產生紀錄\n"
            f"  python -m src.selfplay --smoke-test"
        )

    data = read_log(path)
    iters = data.get("iteration")
    if not iters:
        raise SystemExit(f"{path} 沒有資料。先跑 python -m src.selfplay --smoke-test")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Self-play monitoring", fontsize=14)

    # 左上：和局比例與認輸誤判率（兩個健康指標）
    ax = axes[0][0]
    if "draw_rate" in data:
        ax.plot(iters, data["draw_rate"], marker="o", label="draw rate")
    if "false_resign_rate" in data:
        ax.plot(iters, data["false_resign_rate"], marker="s", label="false resign rate")
    ax.axhline(DRAW_RATE_WARN, color="red", linestyle="--", linewidth=1, label="draw 70% (warn)")
    ax.axhline(FALSE_RESIGN_WARN, color="orange", linestyle=":", linewidth=1, label="resign 5% (warn)")
    ax.set_title("Health indicators")
    ax.set_xlabel("iteration")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 右上：平均步數（認輸機制有沒有在生效）
    ax = axes[0][1]
    if "avg_plies" in data:
        ax.plot(iters, data["avg_plies"], marker="o", color="tab:green")
    ax.set_title("Average game length (plies)")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)

    # 左下：訓練損失
    ax = axes[1][0]
    if "policy_loss" in data:
        ax.plot(iters, data["policy_loss"], marker="o", label="policy loss")
    if "value_loss" in data:
        ax.plot(iters, data["value_loss"], marker="s", label="value loss")
    ax.set_title("Training loss")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 右下：Elo 累積（P21 要看的就是這條有沒有往上）
    ax = axes[1][1]
    elo = data.get("elo")
    if elo:
        cumulative: list[float] = []
        total = 0.0
        for value in elo:
            total += value
            cumulative.append(total)
        ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="tab:purple")
        ax.axhline(0, color="gray", linewidth=1)
    else:
        ax.text(0.5, 0.5, "還沒有 SPRT 結果", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Cumulative Elo vs starting model")
    ax.set_xlabel("gated iteration")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"已存到 {out}")

    if "draw_rate" in data and data["draw_rate"][-1] > DRAW_RATE_WARN:
        print(
            f"[警告] 最新一代的和局比例 {data['draw_rate'][-1] * 100:.0f} % 超過 70 %，"
            f"模型過度保守。\n"
            f"       調高 config.yaml 的 mcts.dirichlet_epsilon 或 mcts.temperature_moves。"
        )


if __name__ == "__main__":
    main()
