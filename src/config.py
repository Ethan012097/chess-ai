"""讀取 config.yaml，轉成 dataclass 方便 IDE 補全與型別檢查。

為什麼用 dataclass 而不是直接傳 dict：dict 打錯 key 只會在執行時才炸，
而且沒有補全。dataclass 在載入當下就會因為缺欄位而報錯，比較好除錯。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 專案根目錄（本檔案在 <root>/src/config.py，所以往上兩層）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class ModelConfig:
    """網路結構參數。"""

    channels: int = 128
    blocks: int = 10
    value_head_channels: int = 8
    value_hidden: int = 256


@dataclass
class TrainConfig:
    """訓練超參數。"""

    batch_size: int = 1024
    epochs: int = 12
    optimizer: str = "adamw"
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    lr_final_factor: float = 0.05
    grad_clip: float = 1.0
    value_weight: float = 1.0
    amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every: int = 100
    eval_every: int = 5000
    eval_batches: int = 50
    checkpoint_every: int = 1
    soft_targets: bool = False


@dataclass
class DataConfig:
    """資料路徑與前處理的篩選 / 取樣規則。"""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_file: str = "data/processed/train.npy"
    val_file: str = "data/processed/val.npy"
    val_ratio: float = 0.02
    min_elo: int = 2000
    min_time_control_seconds: int = 180
    min_game_plies: int = 20
    max_game_plies: int = 300
    skip_opening_plies: int = 8
    skip_ending_plies: int = 2
    max_positions_per_game: int = 40
    shard_size: int = 1_000_000


@dataclass
class SearchConfig:
    """Phase 1 的下棋參數。"""

    temperature: float = 0.0
    top_k: int = 5
    use_mate_check: bool = True
    use_lookahead: bool = True


@dataclass
class EvalConfig:
    """評估參數。"""

    stockfish_path: str = "bin/stockfish.exe"
    match_games: int = 100
    baseline_games: int = 200
    random_opening_plies: int = 4
    skill_levels: list[int] = field(default_factory=lambda: [0, 3, 5])
    engine_movetime_ms: int = 100
    # cutechess-cli（--mode tournament / sprt）
    cutechess_path: str = "bin/cutechess-1.5.1-win64/cutechess-cli.exe"
    engine_bat: str = "engine.bat"
    openings_file: str = "data/openings.pgn"
    tournament_rounds: int = 100
    time_control: str = "10+0.1"
    concurrency: int = 2
    # SPRT
    sprt_elo0: int = 0
    sprt_elo1: int = 20
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05


@dataclass
class Config:
    """整包設定。`presets` 與 Phase 2 的區塊用原始 dict 保存即可。"""

    device: str = "auto"
    preset: str = "base"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcts: dict[str, Any] = field(default_factory=dict)      # Phase 2 預留，Phase 1 不使用
    selfplay: dict[str, Any] = field(default_factory=dict)  # Phase 2 預留，Phase 1 不使用

    def resolve_path(self, relative: str) -> Path:
        """把 config 裡的相對路徑轉成以專案根目錄為基準的絕對路徑。"""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        """轉回巢狀 dict，用來存進 checkpoint。"""
        return {
            "device": self.device,
            "preset": self.preset,
            "seed": self.seed,
            "model": vars(self.model),
            "train": vars(self.train),
            "data": vars(self.data),
            "search": vars(self.search),
            "eval": vars(self.eval),
            "presets": self.presets,
            "mcts": self.mcts,
            "selfplay": self.selfplay,
        }


def _filter_known(section: dict[str, Any] | None, cls: type) -> dict[str, Any]:
    """只挑出 dataclass 認得的欄位，避免 config.yaml 多寫東西就整個炸掉。"""
    if not section:
        return {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(section) - known
    if unknown:
        print(f"[config] 警告：{cls.__name__} 有不認識的欄位，將被忽略：{sorted(unknown)}")
    return {k: v for k, v in section.items() if k in known}


def load_config(
    path: str | Path | None = None,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """讀 config.yaml 並套用 preset。

    Args:
        path: config.yaml 路徑，None 表示用專案根目錄的預設檔。
        preset: 覆寫 yaml 裡的 `preset` 欄位（例如命令列的 --preset small）。
        overrides: 巢狀覆寫，例如 {"train": {"batch_size": 64}}。

    Returns:
        Config，已套用 preset 的 channels / blocks / batch_size。
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"找不到設定檔 {cfg_path}。\n"
            f"請確認你在專案根目錄執行，且 config.yaml 存在。"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(raw.get(section), dict):
                raw[section].update(values)
            else:
                raw[section] = values

    cfg = Config(
        device=raw.get("device", "auto"),
        preset=preset or raw.get("preset", "base"),
        seed=raw.get("seed", 42),
        model=ModelConfig(**_filter_known(raw.get("model"), ModelConfig)),
        train=TrainConfig(**_filter_known(raw.get("train"), TrainConfig)),
        data=DataConfig(**_filter_known(raw.get("data"), DataConfig)),
        search=SearchConfig(**_filter_known(raw.get("search"), SearchConfig)),
        eval=EvalConfig(**_filter_known(raw.get("eval"), EvalConfig)),
        presets=raw.get("presets", {}),
        mcts=raw.get("mcts", {}),
        selfplay=raw.get("selfplay", {}),
    )

    # preset 覆寫模型大小與 batch size
    if cfg.preset:
        if cfg.preset not in cfg.presets:
            raise ValueError(
                f"config.yaml 的 presets 裡沒有 '{cfg.preset}'。"
                f"可用的有：{sorted(cfg.presets)}"
            )
        p = cfg.presets[cfg.preset]
        cfg.model.channels = p.get("channels", cfg.model.channels)
        cfg.model.blocks = p.get("blocks", cfg.model.blocks)
        cfg.train.batch_size = p.get("batch_size", cfg.train.batch_size)

    # overrides 要贏過 preset（命令列的 --batch-size 應該最大）
    if overrides and isinstance(overrides.get("train"), dict):
        for k, v in overrides["train"].items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)

    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """把每支腳本都會用到的共同參數掛上去。"""
    parser.add_argument("--config", type=str, default=None, help="config.yaml 路徑")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["small", "base", "large"],
        help="依 VRAM 選模型大小（覆寫 config.yaml）",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="auto / cuda / cpu（覆寫 config.yaml）"
    )


def main() -> None:
    """`python -m src.config` 會印出目前生效的設定，方便確認 preset 有沒有吃到。"""
    parser = argparse.ArgumentParser(description="印出目前生效的設定")
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device

    print(f"設定檔：{args.config or DEFAULT_CONFIG_PATH}")
    print(f"preset：{cfg.preset}")
    print(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
