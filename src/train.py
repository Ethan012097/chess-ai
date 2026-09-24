"""訓練迴圈。

用法（在專案根目錄執行）：
    python -m src.train --smoke-test          # 只跑 200 步，確認整條路是通的
    python -m src.train --preset small        # 依 VRAM 選 preset
    python -m src.train --resume models/epoch_3.pt

損失函數：
    loss = policy_loss + value_weight * value_loss
    policy_loss = CrossEntropyLoss(policy_logits, move_index)   # 不做 legal mask，
                                                                # 讓網路自己學會合法性
    value_loss  = MSELoss(value.squeeze(), result)

Windows 注意事項：DataLoader 的 worker 會 re-import 主模組，所以進入點**必須**包在
`if __name__ == "__main__":` 裡（本檔案最下面），否則會無限遞迴開行程。
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PROJECT_ROOT, Config, add_common_args, load_config
from src.dataset import build_dataloader
from src.model import ChessNet, resolve_device

SMOKE_TEST_STEPS = 200
LOG_FILE_NAME = "train_log.csv"
# val 指標另外存一份。train_log.csv 的欄位是規格訂死的（只放訓練指標），
# 但 best.pt 是依 val top-1 挑的，跑完之後沒有紀錄就無從回頭判斷有沒有訓歪，
# 所以額外寫這個檔（欄位相同，方便畫在同一張圖上）。
VAL_LOG_FILE_NAME = "val_log.csv"
CSV_COLUMNS = [
    "step",
    "epoch",
    "lr",
    "policy_loss",
    "value_loss",
    "policy_top1",
    "policy_top5",
    "value_mae",
    "positions_per_sec",
]
TOP_K_FOR_METRIC = 5


@dataclass
class Metrics:
    """一段期間內累積的指標，除以樣本數就是平均值。

    用 dataclass 只是為了少寫幾個變數，沒有任何行為。
    """

    policy_loss: float = 0.0
    value_loss: float = 0.0
    top1: float = 0.0
    top5: float = 0.0
    value_abs_err: float = 0.0
    count: int = 0

    def add(
        self,
        policy_loss: float,
        value_loss: float,
        top1: float,
        top5: float,
        value_abs_err: float,
        batch_size: int,
    ) -> None:
        """累加一個 batch 的結果（傳進來的是該 batch 的平均值）。"""
        self.policy_loss += policy_loss * batch_size
        self.value_loss += value_loss * batch_size
        self.top1 += top1 * batch_size
        self.top5 += top5 * batch_size
        self.value_abs_err += value_abs_err * batch_size
        self.count += batch_size

    def average(self) -> dict[str, float]:
        """回傳平均後的指標。count=0 時全部回 0，避免除零。"""
        n = max(self.count, 1)
        return {
            "policy_loss": self.policy_loss / n,
            "value_loss": self.value_loss / n,
            "policy_top1": self.top1 / n,
            "policy_top5": self.top5 / n,
            "value_mae": self.value_abs_err / n,
        }

    def reset(self) -> None:
        self.policy_loss = self.value_loss = self.top1 = self.top5 = 0.0
        self.value_abs_err = 0.0
        self.count = 0


def compute_batch_metrics(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    soft_targets: bool,
) -> tuple[float, float, float]:
    """算 top-1 / top-5 準確率與 value 的 MAE。

    Args:
        policy_logits: (B, 4672)
        value: (B, 1)
        policy_target: soft_targets=False 時是 (B,) int64；True 時是 (B, 4672)
        value_target: (B,)
        soft_targets: policy target 是不是機率向量。

    Returns:
        (top1, top5, value_mae)，都是這個 batch 的平均值。
    """
    with torch.no_grad():
        target_index = policy_target.argmax(dim=1) if soft_targets else policy_target
        topk = policy_logits.topk(TOP_K_FOR_METRIC, dim=1).indices
        correct = topk == target_index.unsqueeze(1)
        top1 = correct[:, 0].float().mean().item()
        top5 = correct.any(dim=1).float().mean().item()
        value_mae = (value.squeeze(-1) - value_target).abs().mean().item()
    return top1, top5, value_mae


def compute_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """算總損失。

    **兩種 policy target 共用同一行程式**（規格 §7.2 要求兩種模式只在損失計算處分岔，
    這裡連分岔都不需要）：

      - `soft_targets: false`（監督式）：target 是 int64 的類別 index，
        `F.cross_entropy` 走一般的分類損失
      - `soft_targets: true`（自我對弈）：target 是 4672 維機率向量，
        `F.cross_entropy` 自動改算 `-(target * log_softmax(logits)).sum(dim=1).mean()`
        —— 正是規格寫的那條公式，而且是融合過的實作，數值上比自己寫更穩

    `tests/test_selfplay.py::test_soft_target_loss_matches_spec_formula` 驗證
    這兩者的數值完全一致，不是「應該一樣」而是真的量過。

    Returns:
        (total_loss, policy_loss, value_loss)
    """
    policy_loss = F.cross_entropy(policy_logits, policy_target)
    value_loss = F.mse_loss(value.squeeze(-1), value_target)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


def make_lr_schedule(cfg: Config, total_steps: int):
    """warmup 之後 cosine 降到 lr * lr_final_factor。

    Args:
        cfg: 設定（用 warmup_steps 與 lr_final_factor）。
        total_steps: 整個訓練的總步數。

    Returns:
        一個吃 step、回傳「lr 倍率」的函式，給 LambdaLR 用。
    """
    warmup = cfg.train.warmup_steps
    final_factor = cfg.train.lr_final_factor

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_factor + (1.0 - final_factor) * cosine

    return lr_lambda


def setup_amp(cfg: Config, device: torch.device) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    """決定混合精度要用 bf16 還是 fp16。

    bf16 不需要 GradScaler，但 GTX 10/16 系列不支援 bf16，就要退回 fp16 + scaler。
    這裡用 `torch.cuda.is_bf16_supported()` 自動偵測。

    Returns:
        (啟用與否, dtype, scaler)。scaler 在 bf16 或 CPU 時是停用狀態。
    """
    if not cfg.train.amp or device.type != "cuda":
        return False, torch.float32, torch.amp.GradScaler("cuda", enabled=False)

    if torch.cuda.is_bf16_supported():
        print("[AMP] 使用 bfloat16（不需要 GradScaler）")
        return True, torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)

    print("[AMP] 這張卡不支援 bfloat16，改用 float16 + GradScaler")
    return True, torch.float16, torch.amp.GradScaler("cuda", enabled=True)


@torch.no_grad()
def evaluate_model(
    model: ChessNet,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, float]:
    """在 val 集上算指標。

    Args:
        max_batches: 只跑前 N 個 batch（訓練途中的期中評估用），None 表示跑完整個 val。

    Returns:
        {"policy_loss", "value_loss", "policy_top1", "policy_top5", "value_mae"}
    """
    model.eval()
    metrics = Metrics()

    iterator = enumerate(loader)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    bar = tqdm(iterator, total=total, desc="驗證", leave=False, unit="batch")

    for i, (boards, policy_target, value_target) in bar:
        if max_batches is not None and i >= max_batches:
            break
        boards = boards.to(device, non_blocking=True)
        policy_target = policy_target.to(device, non_blocking=True)
        value_target = value_target.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            policy_logits, value = model(boards)
            _, policy_loss, value_loss = compute_loss(
                policy_logits.float(),
                value.float(),
                policy_target,
                value_target,
                cfg.train.value_weight,
            )

        top1, top5, value_mae = compute_batch_metrics(
            policy_logits.float(), value.float(), policy_target, value_target,
            cfg.train.soft_targets,
        )
        metrics.add(
            policy_loss.item(), value_loss.item(), top1, top5, value_mae, boards.size(0)
        )

    bar.close()
    model.train()
    return metrics.average()


def save_checkpoint(
    path: Path,
    model: ChessNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    cfg: Config,
    best_top1: float,
) -> None:
    """存 checkpoint，內容要足以完整續訓。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg.to_dict(),
            "best_top1": best_top1,
        },
        path,
    )


def append_csv_row(csv_path: Path, row: dict[str, float]) -> None:
    """把一列指標寫進 logs/train_log.csv（檔案不存在就先寫表頭）。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def train(cfg: Config, args: argparse.Namespace) -> None:
    """主訓練迴圈。

    Args:
        cfg: 設定（已套用 preset 與命令列覆寫）。
        args: 命令列參數（用到 smoke_test / resume）。
    """
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    # 輸入形狀固定（batch × 18 × 8 × 8），開 benchmark 讓 cudnn 挑最快的演算法
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_path = cfg.resolve_path(cfg.data.train_file)
    val_path = cfg.resolve_path(cfg.data.val_file)

    # --smoke-test 只用少量資料，避免光是建 DataLoader 就等很久
    max_train_samples = cfg.train.batch_size * SMOKE_TEST_STEPS if args.smoke_test else None
    max_val_samples = cfg.train.batch_size * 5 if args.smoke_test else None

    train_loader = build_dataloader(
        train_path, cfg, shuffle=True, max_samples=max_train_samples, drop_last=True
    )
    val_loader = build_dataloader(
        val_path, cfg, shuffle=False, max_samples=max_val_samples
    )

    model = ChessNet.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    epochs = 1 if args.smoke_test else cfg.train.epochs
    steps_per_epoch = len(train_loader)
    total_steps = SMOKE_TEST_STEPS if args.smoke_test else steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_schedule(cfg, total_steps))
    amp_enabled, amp_dtype, scaler = setup_amp(cfg, device)

    start_epoch = 0
    global_step = 0
    best_top1 = 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(
                f"找不到 checkpoint：{resume_path}\n"
                f"可用的 checkpoint：{sorted(p.name for p in (PROJECT_ROOT / 'models').glob('*.pt'))}"
            )
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_top1 = checkpoint.get("best_top1", 0.0)
        print(f"[續訓] 從 {resume_path} 繼續，epoch={start_epoch}、step={global_step}")

    # 輸出目錄可以指定。理由：換 preset 重訓時 `epoch_N.pt` 會跟舊 preset 的檔名
    # 撞在一起（epoch_1.pt 就是 epoch_1.pt），直接把先前訓練好的模型蓋掉；
    # 而且 `best.pt` 是依 val top-1 自動覆寫的，**完全沒有經過對局把關**。
    # 分開目錄之後，新舊模型可以先 SPRT 對打，確認比較強才手動換掉 best.pt。
    models_dir = PROJECT_ROOT / (args.model_dir or "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best.pt"
    logs_dir = PROJECT_ROOT / "logs"
    csv_path = logs_dir / LOG_FILE_NAME
    val_csv_path = logs_dir / VAL_LOG_FILE_NAME

    print("=" * 70)
    print(f"preset        : {cfg.preset}（channels={cfg.model.channels}, blocks={cfg.model.blocks}）")
    print(f"參數量        : {model.count_parameters():,}")
    print(f"裝置          : {device}", end="")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"（{props.name}, {props.total_memory / (1 << 30):.1f} GB）")
    else:
        print()
    print(f"train 盤面    : {len(train_loader.dataset):,}")
    print(f"val 盤面      : {len(val_loader.dataset):,}")
    print(f"batch_size    : {cfg.train.batch_size}")
    print(f"每 epoch 步數 : {steps_per_epoch:,}")
    print(f"總步數        : {total_steps:,}")
    print(f"lr            : {cfg.train.lr}（warmup {cfg.train.warmup_steps} 步後 cosine）")
    print(f"num_workers   : {cfg.train.num_workers}")
    if args.smoke_test:
        print(f"** --smoke-test：只跑 {SMOKE_TEST_STEPS} 步 **")
    print("=" * 70)

    model.train()
    window = Metrics()          # log_every 期間的滑動統計
    positions_seen = 0
    window_start = time.perf_counter()
    stop = False

    for epoch in range(start_epoch, epochs):
        bar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for boards, policy_target, value_target in bar:
            optimizer.zero_grad(set_to_none=True)
            try:
                # 搬資料到 GPU 也可能 OOM，所以一起包在 try 裡面
                boards = boards.to(device, non_blocking=True)
                policy_target = policy_target.to(device, non_blocking=True)
                value_target = value_target.to(device, non_blocking=True)

                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    policy_logits, value = model(boards)
                    loss, policy_loss, value_loss = compute_loss(
                        policy_logits.float(),
                        value.float(),
                        policy_target,
                        value_target,
                        cfg.train.value_weight,
                    )

                scaler.scale(loss).backward()
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError as exc:
                raise SystemExit(
                    f"\nCUDA out of memory！\n"
                    f"目前設定：preset={cfg.preset}、batch_size={cfg.train.batch_size}、"
                    f"channels={cfg.model.channels}、blocks={cfg.model.blocks}\n"
                    f"下一步，擇一：\n"
                    f"  1. 換小一號的 preset：python -m src.train --preset "
                    f"{'small' if cfg.preset == 'base' else 'base' if cfg.preset == 'large' else 'small'}\n"
                    f"  2. 把 batch_size 減半：python -m src.train --batch-size "
                    f"{cfg.train.batch_size // 2}\n"
                    f"  3. 關掉其他吃 VRAM 的程式（瀏覽器、遊戲）後重試"
                ) from exc

            scheduler.step()
            global_step += 1
            positions_seen += boards.size(0)

            top1, top5, value_mae = compute_batch_metrics(
                policy_logits.float(), value.float(), policy_target, value_target,
                cfg.train.soft_targets,
            )
            window.add(
                policy_loss.item(), value_loss.item(), top1, top5, value_mae, boards.size(0)
            )

            if global_step % cfg.train.log_every == 0:
                avg = window.average()
                elapsed = time.perf_counter() - window_start
                pos_per_sec = window.count / max(elapsed, 1e-9)
                current_lr = scheduler.get_last_lr()[0]

                bar.set_postfix(
                    {
                        "p_loss": f"{avg['policy_loss']:.3f}",
                        "v_loss": f"{avg['value_loss']:.3f}",
                        "top1": f"{avg['policy_top1'] * 100:.1f}%",
                        "pos/s": f"{pos_per_sec:,.0f}",
                    }
                )
                append_csv_row(
                    csv_path,
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "lr": current_lr,
                        "positions_per_sec": round(pos_per_sec, 1),
                        **avg,
                    },
                )
                window.reset()
                window_start = time.perf_counter()

            # 期中評估（跑 val 子集）
            if cfg.train.eval_every > 0 and global_step % cfg.train.eval_every == 0:
                val_metrics = evaluate_model(
                    model, val_loader, device, cfg, amp_enabled, amp_dtype,
                    max_batches=cfg.train.eval_batches,
                )
                tqdm.write(
                    f"  [step {global_step:,}] val "
                    f"top1={val_metrics['policy_top1'] * 100:.2f}% "
                    f"top5={val_metrics['policy_top5'] * 100:.2f}% "
                    f"value_mae={val_metrics['value_mae']:.4f}"
                )
                append_csv_row(
                    val_csv_path,
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "lr": scheduler.get_last_lr()[0],
                        **val_metrics,
                    },
                )
                if val_metrics["policy_top1"] > best_top1:
                    best_top1 = val_metrics["policy_top1"]
                    save_checkpoint(
                        best_path, model, optimizer, scheduler,
                        epoch, global_step, cfg, best_top1,
                    )
                    tqdm.write(f"  → 新的最佳模型，已存到 {best_path}")

            if args.smoke_test and global_step >= SMOKE_TEST_STEPS:
                stop = True
                break

        bar.close()

        if stop:
            break

        # --- 每個 epoch 結束：完整驗證 + 存 checkpoint ---
        val_metrics = evaluate_model(model, val_loader, device, cfg, amp_enabled, amp_dtype)
        append_csv_row(
            val_csv_path,
            {
                "step": global_step,
                "epoch": epoch + 1,
                "lr": scheduler.get_last_lr()[0],
                **val_metrics,
            },
        )
        print(
            f"\nepoch {epoch + 1} 驗證結果："
            f" policy_loss={val_metrics['policy_loss']:.4f}"
            f" top1={val_metrics['policy_top1'] * 100:.2f}%"
            f" top5={val_metrics['policy_top5'] * 100:.2f}%"
            f" value_mae={val_metrics['value_mae']:.4f}"
        )

        if (epoch + 1) % cfg.train.checkpoint_every == 0:
            path = models_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                path, model, optimizer, scheduler, epoch + 1, global_step, cfg, best_top1
            )
            print(f"已存 checkpoint：{path}")

        if val_metrics["policy_top1"] > best_top1:
            best_top1 = val_metrics["policy_top1"]
            save_checkpoint(
                best_path, model, optimizer, scheduler,
                epoch + 1, global_step, cfg, best_top1,
            )
            print(f"新的最佳模型（top1={best_top1 * 100:.2f}%），已存到 {best_path}")

    # --- 收尾 ---
    if args.smoke_test:
        val_metrics = evaluate_model(
            model, val_loader, device, cfg, amp_enabled, amp_dtype, max_batches=5
        )
        # 存一個 checkpoint，這樣馬上就能試 play.py / evaluate.py
        # （棋力當然很爛，只是用來確認那兩支程式跑得起來）
        smoke_path = models_dir / "smoke_test.pt"
        save_checkpoint(
            smoke_path, model, optimizer, scheduler, 0, global_step, cfg, best_top1
        )

        print(f"\n--smoke-test 完成（{global_step} 步）")
        print(
            f"  val top1={val_metrics['policy_top1'] * 100:.2f}% "
            f"top5={val_metrics['policy_top5'] * 100:.2f}% "
            f"value_mae={val_metrics['value_mae']:.4f}"
        )
        print(f"  訓練指標已寫入 {csv_path}")
        print(f"  checkpoint 已存到 {smoke_path}")
        print("\n如果 policy_loss 有從 ~8.4（= ln 4672）往下掉，代表整條路是通的。")
        print("下一步，開始完整訓練：")
        print(f"  python -m src.train --preset {cfg.preset}")
    else:
        print(f"\n訓練完成。最佳 val top-1：{best_top1 * 100:.2f}%")
        print(f"訓練曲線：{csv_path}")
        print("\n下一步：")
        print("  python scripts/plot_training.py            # 畫訓練曲線")
        print("  python -m src.evaluate --mode baseline     # 對隨機走法（應 > 98%）")
        print("  python -m src.play --mode cli              # 人機對弈")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="訓練 policy + value 雙頭網路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"只跑 {SMOKE_TEST_STEPS} 步，用來確認整條路是通的",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="checkpoint 輸出目錄（預設 models/）。換 preset 重訓時務必指定，"
             "否則會覆蓋掉舊 preset 的 epoch_N.pt，以及未經對局把關的 best.pt",
    )
    parser.add_argument("--resume", type=str, default=None, help="從 checkpoint 續訓")
    parser.add_argument("--epochs", type=int, default=None, help="覆寫 config 的 epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="覆寫 batch_size")
    parser.add_argument("--lr", type=float, default=None, help="覆寫學習率")
    parser.add_argument("--num-workers", type=int, default=None, help="覆寫 DataLoader worker 數")
    args = parser.parse_args()

    overrides: dict[str, dict[str, object]] = {"train": {}}
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["train"]["lr"] = args.lr
    if args.num_workers is not None:
        overrides["train"]["num_workers"] = args.num_workers

    cfg = load_config(args.config, preset=args.preset, overrides=overrides)
    if args.device:
        cfg.device = args.device

    train(cfg, args)


# Windows 的 DataLoader worker 會 re-import 這個檔案，
# 沒有這層保護會無限遞迴開行程。
if __name__ == "__main__":
    main()
