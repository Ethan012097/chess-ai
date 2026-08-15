"""評估模型：準確率、對局測試、Elo 估計。

三種模式：
    python -m src.evaluate --mode accuracy    # val 集上的 top-1 / top-5 / value MAE
    python -m src.evaluate --mode baseline    # 對隨機走法打 200 局（勝率應 > 98%）
    python -m src.evaluate --mode match       # 對 Stockfish 打 100 局

Elo 差用標準公式估計：
    score = (勝 + 0.5 * 和) / 總局數
    Elo_diff = -400 * log10(1 / score - 1)
同時輸出 ±95% 信賴區間。

結果會存成 logs/eval_{timestamp}.json。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import subprocess
from collections import Counter
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess
import chess.engine
import numpy as np
import requests
import torch
from tqdm import tqdm

from src.config import PROJECT_ROOT, Config, add_common_args, load_config
from src.dataset import build_dataloader
from src.encoding import compact_to_board, decode_compact_to_planes
from src.model import ChessNet, resolve_device
from src.search import Searcher
from src.search.greedy import GreedySearcher, RandomSearcher
from src.train import compute_batch_metrics, compute_loss, Metrics

DEFAULT_CHECKPOINT = "models/best.pt"
CONFIDENCE_Z = 1.96          # 95% 信賴區間的 z 值
MAX_GAME_PLIES = 400         # 對局的步數上限，避免無限長的和棋拖住評估
# score 為 0 或 1 時 Elo 公式會發散（log10(0)），所以要夾住。
ELO_CLAMP = 1200.0

# 對手的走法函式：吃一個盤面，回傳要走的著法（隨機走法或 Stockfish 都符合）
MoveFunction = Callable[[chess.Board], chess.Move]


@dataclass
class MatchResult:
    """一組對局的結果。"""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """得分率 = (勝 + 0.5 * 和) / 總局數。"""
        if self.total == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total

    @property
    def win_rate(self) -> float:
        """純勝率（不含和局）。"""
        return self.wins / self.total if self.total else 0.0


def elo_difference(score: float) -> float:
    """從得分率估計 Elo 差。

    Args:
        score: 0..1 的得分率。

    Returns:
        Elo 差，夾在 ±ELO_CLAMP。

    Note:
        **點估計與信賴區間的兩個端點都要走這一支**（規格 §3.2）。
        只夾住點估計是錯的——那會把本來正確的數字壓成錯的，
        而且會讓 `assert lower <= point <= upper` 永遠不響。
    """
    # score 為 0 或 1 時 log10 會發散，夾一個極小的邊界進去
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    raw = -400.0 * math.log10(1.0 / score - 1.0)
    return max(-ELO_CLAMP, min(ELO_CLAMP, raw))


def wilson_interval(
    score: float, n: int, z: float = CONFIDENCE_Z
) -> tuple[float, float]:
    """得分率的 Wilson 信賴區間。

    為什麼不用課本上的 Wald 區間（`score ± z * sqrt(p(1-p)/n)`）：
    得分率逼近 0 或 1 時 Wald 的變異數會塌成 0，區間縮成一個點，
    然後點估計就會跑到區間外面（M7 的 198勝2和0負 就踩到這個）。
    Wilson 區間在極端值附近仍然合理，這是它存在的理由。

    Args:
        score: 得分率 0~1（和局算 0.5 局）。
        n: 總局數。
        z: 常態分位數，1.96 對應 95%。

    Returns:
        (下界, 上界)，都夾在 [0, 1]。n = 0 時回傳 (0, 1)（完全沒資訊）。
    """
    if n <= 0:
        return 0.0, 1.0
    z2 = z * z
    center = (score + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(score * (1 - score) / n + z2 / (4 * n * n)) / (1 + z2 / n)

    # Wilson 區間在數學上一定包含觀測到的得分率（可以代數證明 half >= |center - score|），
    # 但浮點數會有誤差：score=1.0 時上界會算成 0.9999999999999999，
    # 讓「區間包含點估計」這個不變式在最極端的情況下失守。
    # 這裡用 min/max 把它夾回去，修正的純粹是捨入誤差，不是統計上的調整。
    low = max(0.0, min(score, center - half))
    high = min(1.0, max(score, center + half))
    return low, high


def elo_confidence_interval(result: MatchResult) -> tuple[float, float, float]:
    """算 Elo 差與 95% 信賴區間。

    先對**得分率**算 Wilson 區間，再把點估計與上下界各自代進同一個
    `elo_difference`。三個數字走同一條換算路徑，
    `lower <= point <= upper` 才是有意義的不變式（規格 §3.2）。

    Args:
        result: 對局結果。

    Returns:
        (elo, elo_low, elo_high)
    """
    n = result.total
    if n == 0:
        return 0.0, -ELO_CLAMP, ELO_CLAMP

    score = result.score
    low, high = wilson_interval(score, n)
    return elo_difference(score), elo_difference(low), elo_difference(high)


# --- mode: accuracy ---------------------------------------------------------


def evaluate_accuracy(model: ChessNet, cfg: Config, device: torch.device,
                      max_batches: int | None = None) -> dict[str, float]:
    """在 val 集上算 policy top-1 / top-5 與 value MAE。

    Args:
        model: 模型。
        cfg: 設定。
        device: 裝置。
        max_batches: 只跑前 N 個 batch，None 表示跑完整個 val 集。

    Returns:
        指標 dict。
    """
    val_path = cfg.resolve_path(cfg.data.val_file)
    loader = build_dataloader(val_path, cfg, shuffle=False)

    model.eval()
    metrics = Metrics()
    total = len(loader) if max_batches is None else min(max_batches, len(loader))

    with torch.no_grad():
        for i, (boards, policy_target, value_target) in enumerate(
            tqdm(loader, total=total, desc="評估準確率", unit="batch")
        ):
            if max_batches is not None and i >= max_batches:
                break
            boards = boards.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            value_target = value_target.to(device, non_blocking=True)

            policy_logits, value = model(boards)
            _, policy_loss, value_loss = compute_loss(
                policy_logits.float(), value.float(), policy_target, value_target,
                cfg.train.value_weight,
            )
            top1, top5, value_mae = compute_batch_metrics(
                policy_logits.float(), value.float(), policy_target, value_target,
                cfg.train.soft_targets,
            )
            metrics.add(
                policy_loss.item(), value_loss.item(), top1, top5, value_mae, boards.size(0)
            )

    result = metrics.average()
    result["positions"] = metrics.count
    return result


# --- 對局 -------------------------------------------------------------------


def random_opening(board: chess.Board, plies: int, rng: random.Random) -> None:
    """在盤面上隨機走前幾步，製造開局變化（不然每局都一模一樣）。

    Args:
        board: 會被就地修改。
        plies: 要隨機走幾步。
        rng: 亂數產生器。
    """
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            return
        board.push(rng.choice(moves))


def play_one_game(
    ai: Searcher,
    opponent_move_fn: MoveFunction,
    ai_is_white: bool,
    opening_plies: int,
    rng: random.Random,
) -> str:
    """下一局，回傳 AI 視角的結果。

    Args:
        ai: 我們的模型。
        opponent_move_fn: 吃 board 回傳 move 的函式（隨機走法或 Stockfish）。
        ai_is_white: AI 是否執白。
        opening_plies: 開局隨機步數。
        rng: 亂數產生器。

    Returns:
        "win" / "draw" / "loss"（都是 AI 的視角）。
    """
    board = chess.Board()
    random_opening(board, opening_plies, rng)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_GAME_PLIES:
        ai_turn = board.turn == (chess.WHITE if ai_is_white else chess.BLACK)
        move = ai.select_move(board) if ai_turn else opponent_move_fn(board)
        board.push(move)

    if not board.is_game_over(claim_draw=True):
        return "draw"      # 打到步數上限，當和局

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    ai_color = chess.WHITE if ai_is_white else chess.BLACK
    return "win" if outcome.winner == ai_color else "loss"


def run_match(
    ai: Searcher,
    opponent_move_fn: MoveFunction,
    num_games: int,
    opening_plies: int,
    seed: int,
    desc: str,
) -> MatchResult:
    """打 N 局，先後手各半。

    Args:
        ai: 我們的模型。
        opponent_move_fn: 對手的走法函式。
        num_games: 局數。
        opening_plies: 開局隨機步數。
        seed: 亂數種子。
        desc: 進度條說明。

    Returns:
        MatchResult。
    """
    rng = random.Random(seed)
    result = MatchResult()

    bar = tqdm(range(num_games), desc=desc, unit="局")
    for i in bar:
        ai_is_white = i % 2 == 0        # 先後手各半
        outcome = play_one_game(ai, opponent_move_fn, ai_is_white, opening_plies, rng)
        if outcome == "win":
            result.wins += 1
        elif outcome == "draw":
            result.draws += 1
        else:
            result.losses += 1
        bar.set_postfix(勝=result.wins, 和=result.draws, 負=result.losses,
                        得分率=f"{result.score * 100:.1f}%")
    bar.close()
    return result


def format_elo(value: float) -> str:
    """把 Elo 數值格式化；碰到上下限就用 ≥ / ≤ 表示，不要假裝那是點估計。"""
    if value >= ELO_CLAMP:
        return f"≥ +{ELO_CLAMP:.0f}"
    if value <= -ELO_CLAMP:
        return f"≤ -{ELO_CLAMP:.0f}"
    # 加 0.0 是為了把 -0.0 正規化成 0.0，否則 50% 得分率會印出難看的 "-0"
    return f"{value + 0.0:+.0f}"


def print_match_report(title: str, result: MatchResult) -> dict:
    """印出勝負統計與 Elo 估計，並回傳可存成 json 的 dict。"""
    elo, elo_low, elo_high = elo_confidence_interval(result)
    # 不變式：點估計一定要落在信賴區間裡面。之前這裡出過 "+920（CI: +768 ~ +800）"
    # 的矛盾輸出，加個 assert 免得以後又改壞。
    assert elo_low <= elo <= elo_high, (
        f"Elo 點估計 {elo} 落在信賴區間 [{elo_low}, {elo_high}] 外面"
    )
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"  局數    : {result.total}")
    print(f"  勝/和/負: {result.wins} / {result.draws} / {result.losses}")
    print(f"  勝率    : {result.win_rate * 100:.1f}%")
    print(f"  得分率  : {result.score * 100:.1f}%")
    print(
        f"  Elo 差  : {format_elo(elo)}"
        f"（95% CI: {format_elo(elo_low)} ~ {format_elo(elo_high)}）"
    )
    return {
        "title": title,
        "games": result.total,
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "score": result.score,
        "elo_diff": elo,
        "elo_ci_low": elo_low,
        "elo_ci_high": elo_high,
    }


# --- mode: baseline ---------------------------------------------------------


def evaluate_baseline(ai: Searcher, cfg: Config, num_games: int | None = None) -> dict:
    """對隨機合法著法打 N 局。這是最低門檻，勝率應該 > 98%。"""
    games = num_games or cfg.eval.baseline_games
    opponent = RandomSearcher(seed=cfg.seed)

    result = run_match(
        ai,
        opponent.select_move,
        games,
        cfg.eval.random_opening_plies,
        cfg.seed,
        "對隨機走法",
    )
    report = print_match_report("baseline：對隨機合法著法", result)

    if result.win_rate > 0.98:
        print("\n  ✓ 通過（勝率 > 98%）")
    else:
        print(
            f"\n  ✗ 未通過（勝率 {result.win_rate * 100:.1f}%，門檻 98%）\n"
            f"  可能原因：訓練不足，或著法編碼 / 鏡射寫錯。\n"
            f"  下一步：python -m pytest tests/test_encoding.py"
        )
    return report


# --- mode: match ------------------------------------------------------------


def stockfish_opponents(
    skill_levels: list[int] | None, uci_elos: list[int] | None
) -> list[tuple[str, dict, int | None]]:
    """組出要對打的 Stockfish 設定清單。

    **兩種限制強度的方式，差別很大：**

    - `Skill Level`（0–20）：靠**在多個候選著法之間隨機挑**來變弱。
      它不是校準過的 Elo 刻度，Stockfish 官方也沒宣稱它是。
      下出來的棋風是「大部分好棋、偶爾莫名其妙送一子」，
      勝率換算成 Elo 並不線性。
    - `UCI_LimitStrength` + `UCI_Elo`：**官方校準過的刻度**（Stockfish 18 是
      1320–3190）。要量「我方大概幾 Elo」就該用這個。

    Args:
        skill_levels: Skill Level 清單，None 表示不用這種。
        uci_elos: UCI_Elo 清單，None 表示不用這種。

    Returns:
        [(標籤, UCI option dict, 已知 Elo 或 None)]。
        第三個元素是「這個對手值多少 Elo」—— 只有 UCI_Elo 那種才知道，
        Skill Level 那種填 None（因為它沒有校準過的對應值）。
    """
    opponents: list[tuple[str, dict, int | None]] = []
    for skill in skill_levels or []:
        # UCI_LimitStrength 要明確關掉，否則若引擎預設開著會蓋掉 Skill Level
        opponents.append(
            (f"Skill Level {skill}", {"UCI_LimitStrength": False, "Skill Level": skill}, None)
        )
    for elo in uci_elos or []:
        # 先開 LimitStrength 再設 Elo（dict 保持插入順序）
        opponents.append(
            (f"UCI_Elo {elo}", {"UCI_LimitStrength": True, "UCI_Elo": elo}, elo)
        )
    return opponents


def check_uci_elo_range(engine: chess.engine.SimpleEngine, wanted: list[int]) -> None:
    """確認要求的 UCI_Elo 落在這顆引擎支援的範圍內。

    超出範圍時 Stockfish 會默默夾住，量出來的數字就變成假的 —— 必須當場擋下來。
    """
    option = engine.options.get("UCI_Elo")
    if option is None:
        raise SystemExit(
            "這個引擎沒有 UCI_Elo 選項，無法用校準刻度評估。\n"
            "請改用 --skill-levels，或換一顆較新的 Stockfish。"
        )
    low, high = option.min, option.max
    bad = [e for e in wanted if e < low or e > high]
    if bad:
        raise SystemExit(
            f"要求的 UCI_Elo {bad} 超出這顆引擎的範圍（{low}–{high}）。\n"
            f"Stockfish 會默默把它夾住，量出來的結果會是錯的。\n"
            f"請改用 {low}–{high} 之間的值。"
        )
    print(f"引擎支援的 UCI_Elo 範圍：{low}–{high}")


# Stockfish 偶爾會有一步卡住不回 bestmove。等這麼久還沒回就當它掛了。
# 設得比 python-chess 預設的 10 秒寬鬆，避免只是機器忙就誤判。
ENGINE_TIMEOUT_S = 30.0
# 同一步最多重開幾次引擎
MAX_ENGINE_RESTARTS = 3


class StockfishMover:
    """包住一個 Stockfish 行程，卡住就自動重開。

    這裡開 class 的理由：`run_match` 要的是一個「給盤面回著法」的函式，
    但那個函式背後需要一個**可以被替換掉**的引擎行程。用 class 存狀態最直接。

    為什麼需要它：實測跑 450 局的評估時，Stockfish 有一步卡了四分多鐘不回應，
    python-chess 丟出 TimeoutError，**整批測量就這樣沒了**。
    一次 hiccup 不該讓幾十分鐘的結果歸零。
    """

    def __init__(self, engine_path: Path, options: dict, movetime_ms: int) -> None:
        self.engine_path = engine_path
        self.options = options
        self.limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        self.restarts = 0
        self.engine = self._open()

    def _open(self) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(
            str(self.engine_path), timeout=ENGINE_TIMEOUT_S
        )
        engine.configure(self.options)
        return engine

    def play(self, board: chess.Board) -> chess.Move:
        """讓 Stockfish 走一步；引擎沒回應就重開再試。

        Raises:
            RuntimeError: 重開 MAX_ENGINE_RESTARTS 次都失敗。
        """
        for attempt in range(MAX_ENGINE_RESTARTS + 1):
            try:
                move = self.engine.play(board, self.limit).move
                if move is not None:
                    return move
                raise chess.engine.EngineError("引擎回傳了空著法")
            except (TimeoutError, chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
                if attempt >= MAX_ENGINE_RESTARTS:
                    raise RuntimeError(
                        f"Stockfish 連續 {MAX_ENGINE_RESTARTS} 次無法回應（{type(exc).__name__}）。\n"
                        f"盤面：{board.fen()}\n"
                        f"下一步：確認 {self.engine_path} 能正常執行，"
                        f"或把 config.yaml 的 eval.engine_movetime_ms 調大。"
                    ) from exc
                self.restarts += 1
                print(
                    f"\n  [警告] Stockfish 沒有回應（{type(exc).__name__}），重開引擎"
                    f"（第 {self.restarts} 次）。這一步會重走。"
                )
                self.close()
                self.engine = self._open()
        raise AssertionError("不會走到這裡")

    def close(self) -> None:
        """關掉引擎行程。已經死掉的話就無視。"""
        try:
            self.engine.quit()
        except Exception:      # noqa: BLE001 — 收屍就是要吃掉所有例外
            pass


def evaluate_match(
    ai: Searcher,
    cfg: Config,
    opponents: list[tuple[str, dict, int | None]],
    num_games: int | None = None,
) -> list[dict]:
    """對 Stockfish 打 N 局，每種強度設定各打一組。

    Args:
        ai: 我方的 searcher。
        cfg: 設定。
        opponents: `stockfish_opponents()` 的輸出。
        num_games: 每組打幾局，None 表示讀 config。

    Returns:
        每組一份報告 dict，多了 `opponent_elo` 欄位（不知道就是 None）。
    """
    games = num_games or cfg.eval.match_games
    engine_path = cfg.resolve_path(cfg.eval.stockfish_path)

    if not engine_path.exists():
        raise SystemExit(
            f"找不到 Stockfish：{engine_path}\n"
            f"下一步：\n"
            f"  1. 到 https://stockfishchess.org/download/ 下載 Windows 版\n"
            f"  2. 解壓後把執行檔放到 {engine_path}\n"
            f"  3. 或修改 config.yaml 的 eval.stockfish_path\n"
            f"（不想裝 Stockfish 的話，可以先跑 --mode baseline）"
        )

    wanted_elos = [elo for _, _, elo in opponents if elo is not None]
    if wanted_elos:
        with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as probe:
            check_uci_elo_range(probe, wanted_elos)

    reports: list[dict] = []
    for index, (label, options, opponent_elo) in enumerate(opponents):
        print(f"\n--- Stockfish {label} ---")
        mover = StockfishMover(engine_path, options, cfg.eval.engine_movetime_ms)
        try:
            result = run_match(
                ai,
                mover.play,
                games,
                cfg.eval.random_opening_plies,
                cfg.seed + index,
                f"對 Stockfish {label}",
            )
        finally:
            mover.close()
        if mover.restarts:
            print(f"  （過程中重開引擎 {mover.restarts} 次，見上方警告）")
        report = print_match_report(f"對 Stockfish {label}", result)
        report["opponent_elo"] = opponent_elo
        reports.append(report)

        if options.get("Skill Level") == 0 and result.wins > 0:
            print("\n  ✓ 對 Skill Level 0 有勝場（M9 達成）")

    if wanted_elos:
        print_absolute_elo(reports)
    return reports


def print_absolute_elo(reports: list[dict]) -> None:
    """用 UCI_Elo 對手當標尺，推估我方的絕對 Elo。

    每個對手給一個獨立估計：`我方 Elo = 對手 Elo + Elo差(得分率)`。
    信賴區間直接沿用該場的區間平移過去。

    **不做加權平均。** 幾個估計值如果彼此差很多，那代表模型的棋力
    在不同對手強度下表現不一致（很常見），硬平均只會把這個訊號抹掉。
    並排列出來、讓人自己看哪些一致，比一個假的單一數字誠實。
    """
    anchored = [r for r in reports if r.get("opponent_elo") is not None]
    if not anchored:
        return

    print(f"\n{'=' * 60}")
    print("絕對 Elo 推估（以 Stockfish 的 UCI_Elo 校準刻度為標尺）")
    print(f"{'=' * 60}")
    print(f"  {'對手 Elo':>9}{'得分率':>9}{'我方 Elo':>11}{'95 % 區間':>20}")

    estimates: list[float] = []
    for r in anchored:
        anchor = r["opponent_elo"]
        point = anchor + r["elo_diff"]
        low = anchor + r["elo_ci_low"]
        high = anchor + r["elo_ci_high"]
        estimates.append(point)
        print(
            f"  {anchor:>9}{r['score'] * 100:>8.1f}%{point:>11.0f}"
            f"{f'[{low:.0f}, {high:.0f}]':>20}"
        )

    spread = max(estimates) - min(estimates)
    print(f"\n  幾個估計之間相差 {spread:.0f} Elo。")
    if spread > 150:
        print(
            "  差距偏大，代表棋力隨對手強度變化明顯（例如打得贏弱手但被強手輾壓）。\n"
            "  這種情況下單一數字沒有意義，要看你關心的是哪個區間。"
        )
    else:
        print(f"  彼此相當一致，可以說「大約 {sum(estimates) / len(estimates):.0f} Elo」。")
    print(
        "\n  注意：這是 Stockfish 自己的校準刻度（大致對齊 CCRL/FIDE），\n"
        "  官方也說只是近似。Lichess 的分數通常比這個高 200–400。"
    )


# --- mode: value-quality ----------------------------------------------------

VALUE_QUALITY_POSITIONS = 3000
VALUE_QUALITY_DEPTH = 12
# Spearman 低於這個值就不要進第 6 節的 MCTS（規格 §5.2）
SPEARMAN_GATE = 0.60
# 重訓 value head 之後的目標（規格 §5.5）
SPEARMAN_TARGET = 0.75
# Stockfish 回報將死時換算成的 centipawn，避免 inf 汙染統計
MATE_CP = 10000


def rank_data(values: np.ndarray) -> np.ndarray:
    """把數值轉成名次，同分取平均名次（Spearman 需要）。

    自己寫是為了不要為了一個函式引入 scipy。

    Args:
        values: 一維陣列。

    Returns:
        同長度的名次陣列（1-based，同分取平均）。
    """
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    # 處理同分：把同一組的名次改成平均值
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0    # 名次是 1-based
        i = j + 1
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 等級相關係數 = 名次上的 Pearson 相關係數。

    為什麼看等級相關而不是絕對誤差：MCTS 需要的是**盤面排序能力**
    （哪個局面比較好），不是絕對數值準不準（規格 §5.2）。

    Args:
        x, y: 等長的一維陣列。

    Returns:
        -1 ~ 1 的相關係數；樣本不足或無變異時回傳 0。
    """
    if len(x) < 2:
        return 0.0
    rx, ry = rank_data(x), rank_data(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denominator == 0:
        return 0.0
    return float((rx * ry).sum() / denominator)


def evaluate_value_quality(
    model: ChessNet,
    cfg: Config,
    device: torch.device,
    num_positions: int = VALUE_QUALITY_POSITIONS,
    depth: int = VALUE_QUALITY_DEPTH,
) -> dict:
    """用 Stockfish 當基準，量測 value head 的**盤面排序能力**。

    為什麼不看 MAE：MAE 對「最終誰贏」這種標籤沒有有意義的下界。假設某盤面真實
    勝率是 0.6，完美的評估器會輸出 +0.2，但標籤只會是 +1 或 -1，誤差恆為 0.8 或
    1.2。**就算把 Stockfish 本人當 value head，對這個 val 集算出的 MAE 也不會低
    到 0.5**（規格 §5.1）。所以這裡改量 Spearman 等級相關係數與正負號一致率。

    Returns:
        {"spearman", "sign_agreement", "positions", "depth", ...}
    """
    val_path = cfg.resolve_path(cfg.data.val_file)
    if not val_path.exists():
        raise SystemExit(
            f"找不到 {val_path}\n"
            f'下一步：python -m src.preprocess --input "data/raw/*.pgn"'
        )
    engine_path = cfg.resolve_path(cfg.eval.stockfish_path)
    if not engine_path.exists():
        raise SystemExit(
            f"找不到 Stockfish：{engine_path}\n"
            f"請到 https://stockfishchess.org/download/ 下載並放到 bin\\stockfish.exe"
        )

    data = np.load(val_path, mmap_mode="r")
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(data), size=min(num_positions, len(data)), replace=False)
    indices.sort()      # 排序過的 index 讀 memmap 比較快

    print(f"從 {val_path.name} 抽 {len(indices):,} 個盤面")
    print(f"Stockfish depth {depth}（這段會跑一陣子）\n")

    model.eval()
    model_values: list[float] = []
    engine_cps: list[float] = []
    skipped = 0

    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        limit = chess.engine.Limit(depth=depth)
        for idx in tqdm(indices, desc="評估", unit="盤面"):
            row = data[int(idx)]
            board = compact_to_board(
                row["pieces"], int(row["castling"]), int(row["ep_square"]),
                int(row["halfmove"]),
            )
            # 還原出來的是 canonical 盤面（白方走棋），已結束的局面沒有評估意義
            if board.is_game_over() or not board.is_valid():
                skipped += 1
                continue

            try:
                info = engine.analyse(board, limit)
                # canonical 盤面一律白方走棋，所以 white pov == 走棋方 pov，
                # 跟 value head 的視角一致，不用轉換
                score = info["score"].white()
                cp = score.score(mate_score=MATE_CP)
            except (chess.engine.EngineError, KeyError, ValueError):
                skipped += 1
                continue
            if cp is None:
                skipped += 1
                continue

            planes = decode_compact_to_planes(
                row["pieces"], int(row["castling"]), int(row["ep_square"]),
                int(row["halfmove"]),
            )
            with torch.no_grad():
                tensor = torch.from_numpy(planes).unsqueeze(0).to(device)
                _, value = model(tensor)

            model_values.append(float(value.item()))
            engine_cps.append(float(cp))

    if len(model_values) < 2:
        raise SystemExit("有效樣本太少，無法計算相關係數。")

    mv = np.array(model_values)
    ec = np.array(engine_cps)

    spearman = spearman_correlation(mv, ec)
    # 正負號一致率：雙方對「誰佔優」的判斷是否一致。
    # 兩邊都判為均勢（接近 0）也算一致。
    model_sign = np.sign(mv)
    engine_sign = np.sign(ec)
    sign_agreement = float((model_sign == engine_sign).mean())

    print(f"\n{'=' * 60}")
    print("value head 品質（對照 Stockfish）")
    print(f"{'=' * 60}")
    print(f"  有效盤面      : {len(mv):,}（跳過 {skipped}）")
    print(f"  Stockfish 深度: {depth}")
    print(f"  Spearman 相關 : {spearman:.4f}")
    print(f"  正負號一致率  : {sign_agreement * 100:.1f}%")
    print()
    if spearman >= SPEARMAN_TARGET:
        print(f"  ✓ 達到重訓後的目標（>= {SPEARMAN_TARGET}）")
    elif spearman >= SPEARMAN_GATE:
        print(
            f"  ✓ 高於進入 MCTS 的門檻（>= {SPEARMAN_GATE}），可以進第 6 節。\n"
            f"    但仍低於重訓目標 {SPEARMAN_TARGET}，做第 5 節的 value 重訓會更好。"
        )
    else:
        print(
            f"  ✗ 低於門檻 {SPEARMAN_GATE}，**先不要進第 6 節的 MCTS**。\n"
            f"    MCTS 對 value 品質極度敏感——value 是每次模擬的葉節點評估，\n"
            f"    歪了會讓搜尋系統性地往錯誤方向展開，比不搜尋還糟。\n"
            f"    下一步：用 Stockfish 評分重訓 value head（規格 §5.4 / §5.5）"
        )

    return {
        "spearman": spearman,
        "sign_agreement": sign_agreement,
        "positions": len(mv),
        "skipped": skipped,
        "depth": depth,
        "gate": SPEARMAN_GATE,
        "target": SPEARMAN_TARGET,
    }


# --- mode: puzzles ----------------------------------------------------------

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
# Rating 分桶邊界：<1000、1000–1200、…、2000+
PUZZLE_RATING_EDGES = (1000, 1200, 1400, 1600, 1800, 2000)
PUZZLE_DEFAULT_COUNT = 2000


def puzzle_bucket_name(rating: int) -> str:
    """把 Rating 對應到分桶名稱。"""
    if rating < PUZZLE_RATING_EDGES[0]:
        return f"<{PUZZLE_RATING_EDGES[0]}"
    for low, high in zip(PUZZLE_RATING_EDGES, PUZZLE_RATING_EDGES[1:]):
        if low <= rating < high:
            return f"{low}-{high}"
    return f"{PUZZLE_RATING_EDGES[-1]}+"


def all_puzzle_buckets() -> list[str]:
    """所有分桶名稱，由易到難。"""
    names = [f"<{PUZZLE_RATING_EDGES[0]}"]
    names += [
        f"{low}-{high}"
        for low, high in zip(PUZZLE_RATING_EDGES, PUZZLE_RATING_EDGES[1:])
    ]
    names.append(f"{PUZZLE_RATING_EDGES[-1]}+")
    return names


def iter_puzzle_rows(local_file: Path | None = None):
    """串流讀取 lichess 謎題資料庫，一列一列吐出。

    整個檔案壓縮後就有 300 MB 以上，**一定要串流**。而且分桶很快就會滿，
    通常只會實際下載前面幾 MB 就停了。

    Args:
        local_file: 本機的 .csv.zst；None 表示直接從網路串流。

    Yields:
        csv.DictReader 的每一列。
    """
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    if local_file is not None and local_file.exists():
        source = open(local_file, "rb")
        stream = source
    else:
        response = requests.get(
            PUZZLE_URL, stream=True, timeout=90,
            headers={"User-Agent": "chess-ai puzzle evaluator"},
        )
        response.raise_for_status()
        source = response
        stream = response.raw

    try:
        with dctx.stream_reader(stream) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            yield from csv.DictReader(text)
    finally:
        source.close()


def collect_puzzles(
    count: int, local_file: Path | None = None
) -> dict[str, list[dict]]:
    """依 Rating 分桶收集謎題，每桶收滿就停。

    Args:
        count: 總共要幾題。
        local_file: 本機檔案，None 表示從網路串流。

    Returns:
        {分桶名稱: [謎題, ...]}
    """
    buckets = all_puzzle_buckets()
    per_bucket = max(count // len(buckets), 1)
    collected: dict[str, list[dict]] = {name: [] for name in buckets}

    print(f"收集謎題（每桶 {per_bucket} 題，共 {per_bucket * len(buckets)} 題）…")
    bar = tqdm(total=per_bucket * len(buckets), desc="收集", unit="題")
    for row in iter_puzzle_rows(local_file):
        try:
            rating = int(row["Rating"])
        except (KeyError, ValueError):
            continue
        name = puzzle_bucket_name(rating)
        if len(collected[name]) < per_bucket:
            collected[name].append(row)
            bar.update(1)
            if all(len(v) >= per_bucket for v in collected.values()):
                break
    bar.close()
    return collected


def solve_puzzle(searcher: Searcher, row: dict) -> bool | None:
    """解一題謎題，回傳是否答對。

    **關鍵細節**（規格 §3.5）：CSV 裡的 `FEN` 是**對手走之前**的盤面，
    要先 push `Moves` 的第一步，才得到真正要解的局面。
    直接拿 FEN 去解會整批答錯，而且錯得很難察覺。

    Args:
        searcher: 要測試的搜尋器。
        row: CSV 的一列。

    Returns:
        True/False 表示答對與否；資料有問題時回傳 None（該題跳過）。
    """
    try:
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        if len(moves) < 2:
            return None
        # 第一步是對手走的，push 完才輪到我們解題
        board.push_uci(moves[0])
        expected = chess.Move.from_uci(moves[1])
        if expected not in board.legal_moves:
            return None
        return searcher.select_move(board) == expected
    except (ValueError, KeyError, AssertionError):
        return None


def solve_puzzle_detailed(searcher: Searcher, row: dict) -> dict | None:
    """解一題謎題，回傳完整過程（配對比較與失敗分析用）。

    跟 `solve_puzzle` 的差別只在回傳內容：這支還會附上盤面、正解、實際選擇、
    以及（MCTS 才有的）訪問次數分佈。要查「為什麼答錯」就需要這些。

    Returns:
        {"correct", "fen", "expected", "chosen", "visits"}；資料有問題時回 None。
    """
    try:
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        if len(moves) < 2:
            return None
        board.push_uci(moves[0])
        expected = chess.Move.from_uci(moves[1])
        if expected not in board.legal_moves:
            return None
        chosen = searcher.select_move(board)
        # MCTS 才有訪問次數；greedy 沒有這個方法，用 getattr 判斷比 isinstance 好，
        # 這樣以後多一種 searcher 也不用改這裡
        visit_fn = getattr(searcher, "visit_counts", None)
        visits = visit_fn(board) if visit_fn is not None else None
        return {
            "correct": chosen == expected,
            "fen": board.fen(),
            "expected": board.san(expected),
            "expected_uci": expected.uci(),
            "chosen": board.san(chosen),
            "chosen_uci": chosen.uci(),
            "visits": visits,
        }
    except (ValueError, KeyError, AssertionError):
        return None


def mcnemar_exact_p(b: int, c: int) -> float:
    """McNemar 檢定的雙尾精確 p 值。

    **為什麼要用配對檢定**：兩個搜尋器跑的是**同一批題目**，
    大部分題目兩邊都答對或都答錯，那些完全不帶訊息。
    只有「一個對、一個錯」的**不一致對**才是證據。

    用獨立樣本的標準誤（每桶 100 題 → 約 ±5 %）會嚴重高估雜訊：
    100 題裡若只有 12 對不一致，判斷依據是那 12 對，不是 100 題。

    Args:
        b: A 答對而 B 答錯的題數。
        c: A 答錯而 B 答對的題數。

    Returns:
        雙尾 p 值。b + c = 0（完全一致）時回傳 1.0。

    用精確二項檢定而不是卡方近似：不一致對常常只有個位數，
    卡方在小樣本下會給出過度樂觀的 p 值。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def evaluate_puzzles_paired(
    searcher_a: Searcher,
    searcher_b: Searcher,
    label_a: str,
    label_b: str,
    count: int,
    local_file: Path | None = None,
    show_failures: int = 10,
) -> dict:
    """在**同一批題目**上跑兩個搜尋器，用 McNemar 檢定比較。

    Args:
        searcher_a / searcher_b: 兩個要比較的搜尋器。
        label_a / label_b: 顯示用的名稱。
        count: 每桶題數。
        local_file: 謎題 csv。
        show_failures: 每桶列出幾題「A 對 B 錯」的細節（0 = 不列）。

    Returns:
        每桶的 {a_correct, b_correct, b_count, c_count, p_value, regressions}。
    """
    collected = collect_puzzles(count, local_file)
    results: dict[str, dict] = {}

    print(f"\n配對比較：{label_a}  vs  {label_b}（同一批題目）\n")
    header = f"  {'Rating':>10} {label_a[:8]:>9} {label_b[:8]:>9}   {'b':>4} {'c':>4} {'p 值':>8}  判定"
    print(header)
    print("  " + "-" * (len(header) - 2))

    totals = {"a": 0, "b": 0, "n": 0, "b_count": 0, "c_count": 0}
    for name in all_puzzle_buckets():
        rows = collected[name]
        if not rows:
            continue

        a_correct = b_correct = attempted = 0
        only_a = only_b = 0
        regressions: list[dict] = []

        for row in tqdm(rows, desc=f"Rating {name:>10}", unit="題", leave=False):
            ra = solve_puzzle_detailed(searcher_a, row)
            rb = solve_puzzle_detailed(searcher_b, row)
            if ra is None or rb is None:
                continue
            attempted += 1
            a_correct += int(ra["correct"])
            b_correct += int(rb["correct"])
            if ra["correct"] and not rb["correct"]:
                only_a += 1
                if len(regressions) < show_failures:
                    regressions.append(
                        {
                            "fen": rb["fen"],
                            "expected": rb["expected"],
                            "chosen": rb["chosen"],
                            "chosen_uci": rb["chosen_uci"],
                            "visits": rb["visits"],
                        }
                    )
            elif rb["correct"] and not ra["correct"]:
                only_b += 1

        p = mcnemar_exact_p(only_a, only_b)
        verdict = "顯著" if p < 0.05 else ("邊緣" if p < 0.10 else "不顯著")
        direction = ""
        if p < 0.10:
            direction = f"（{label_b} 較{'差' if only_a > only_b else '好'}）"

        print(
            f"  {name:>10} {a_correct / max(attempted, 1) * 100:>8.1f}% "
            f"{b_correct / max(attempted, 1) * 100:>8.1f}%   "
            f"{only_a:>4} {only_b:>4} {p:>8.4f}  {verdict}{direction}"
        )

        results[name] = {
            "attempted": attempted,
            "a_correct": a_correct,
            "b_correct": b_correct,
            "b_count": only_a,
            "c_count": only_b,
            "p_value": p,
            "regressions": regressions,
        }
        totals["a"] += a_correct
        totals["b"] += b_correct
        totals["n"] += attempted
        totals["b_count"] += only_a
        totals["c_count"] += only_b

    p_all = mcnemar_exact_p(totals["b_count"], totals["c_count"])
    print(
        f"  {'總計':>10} {totals['a'] / max(totals['n'], 1) * 100:>8.1f}% "
        f"{totals['b'] / max(totals['n'], 1) * 100:>8.1f}%   "
        f"{totals['b_count']:>4} {totals['c_count']:>4} {p_all:>8.4f}"
    )
    print(
        f"\n  b = {label_a} 對而 {label_b} 錯的題數；c = 反過來。\n"
        f"  只有這些**不一致對**帶訊息；兩邊都對或都錯的題目不影響檢定。"
    )

    results["_overall"] = {
        "attempted": totals["n"],
        "a_correct": totals["a"],
        "b_correct": totals["b"],
        "b_count": totals["b_count"],
        "c_count": totals["c_count"],
        "p_value": p_all,
    }
    print_puzzle_regressions(results, label_a, label_b, show_failures)
    return results


def print_puzzle_regressions(
    results: dict, label_a: str, label_b: str, limit: int
) -> None:
    """列出「A 答對但 B 答錯」的題目細節，用來查退步的原因。"""
    if limit <= 0:
        return
    for name, data in results.items():
        if name.startswith("_") or not data.get("regressions"):
            continue
        print(f"\n{'=' * 74}")
        print(f"Rating {name}：{label_a} 答對但 {label_b} 答錯的前 {len(data['regressions'])} 題")
        print(f"{'=' * 74}")
        for i, r in enumerate(data["regressions"], 1):
            print(f"\n  [{i}] {r['fen']}")
            print(f"      正解 {r['expected']}   實際走 {r['chosen']}")
            if r["visits"]:
                top = sorted(r["visits"].items(), key=lambda kv: kv[1], reverse=True)[:6]
                total = sum(r["visits"].values())
                spread = " ".join(f"{u}:{n}" for u, n in top)
                print(f"      訪問次數（共 {total}）: {spread}")


def evaluate_puzzles(
    searcher: Searcher, count: int, local_file: Path | None = None
) -> dict:
    """跑 lichess 謎題測驗，輸出各分數桶的命中率。

    這組數字是最能對外說明成果的指標，也是之後驗證 MCTS 有沒有用的對照組
    —— 同一批題目接上搜尋再跑一次就知道。

    **心理準備**：純 policy 網路在深度戰術上會很難看，四步組合殺基本上是猜。
    這正是 MCTS 要解決的問題。
    """
    collected = collect_puzzles(count, local_file)

    results: dict[str, dict] = {}
    total_correct = total_attempted = 0

    print()
    for name in all_puzzle_buckets():
        rows = collected[name]
        if not rows:
            continue
        correct = attempted = 0
        for row in tqdm(rows, desc=f"Rating {name:>10}", unit="題", leave=False):
            outcome = solve_puzzle(searcher, row)
            if outcome is None:
                continue
            attempted += 1
            correct += int(outcome)

        accuracy = correct / attempted if attempted else 0.0
        results[name] = {
            "correct": correct,
            "attempted": attempted,
            "accuracy": accuracy,
        }
        total_correct += correct
        total_attempted += attempted
        bar_width = int(accuracy * 40)
        print(
            f"  Rating {name:>10}: {accuracy * 100:5.1f}%  "
            f"({correct:>4}/{attempted:<4}) {'█' * bar_width}"
        )

    overall = total_correct / total_attempted if total_attempted else 0.0
    print(f"\n  總計: {overall * 100:.1f}%（{total_correct}/{total_attempted}）")
    print(
        "\n  註：純 policy 網路在深度戰術上本來就會很難看，高分桶答對率低是正常的。\n"
        "      這正是 MCTS 要解決的問題 —— 接上搜尋之後再跑一次同一批題目對照。"
    )

    return {"buckets": results, "overall": overall, "attempted": total_attempted}


# --- mode: tournament / sprt（透過 cutechess-cli）---------------------------


def resolve_cutechess(cfg: Config) -> Path:
    """找出 cutechess-cli 的位置，找不到就給安裝指引。"""
    path = cfg.resolve_path(cfg.eval.cutechess_path)
    if path.exists():
        return path

    # 版本號可能不同，掃一下 bin/ 底下有沒有其他版本
    bin_dir = PROJECT_ROOT / "bin"
    if bin_dir.exists():
        found = sorted(bin_dir.glob("cutechess*/cutechess-cli.exe"))
        if found:
            print(f"[提醒] config 指到 {path.name} 不存在，改用 {found[0]}")
            return found[0]

    raise SystemExit(
        f"找不到 cutechess-cli：{path}\n"
        f"下一步：\n"
        f"  1. 到 https://github.com/cutechess/cutechess/releases 下載 win64.zip\n"
        f"     （免安裝可攜版，同一包裡有 GUI 與 cutechess-cli）\n"
        f"  2. 解壓到 bin\\，例如 bin\\cutechess-1.5.1-win64\\\n"
        f"  3. 或修改 config.yaml 的 eval.cutechess_path\n"
        f"（不想裝的話，可以先跑 --mode match，那個只需要 python-chess）"
    )


def build_cutechess_command(
    cfg: Config,
    engine_a: list[str],
    engine_b: list[str],
    rounds: int,
    pgn_out: Path,
    sprt: tuple[int, int, float, float] | None = None,
) -> list[str]:
    """組出 cutechess-cli 的指令。

    Args:
        cfg: 設定。
        engine_a: 第一個引擎的 `-engine ...` 參數（不含 `-engine` 本身）。
        engine_b: 第二個引擎的參數。
        rounds: 要打幾輪。
        pgn_out: 對局記錄輸出路徑。
        sprt: (elo0, elo1, alpha, beta)，None 表示不做序貫檢定。

    Returns:
        可以丟給 subprocess 的參數列。
    """
    cutechess = resolve_cutechess(cfg)
    openings = cfg.resolve_path(cfg.eval.openings_file)

    command = [str(cutechess), "-engine", *engine_a, "-engine", *engine_b]
    # -each 的設定會套用到兩個引擎
    command += ["-each", "proto=uci", f"tc={cfg.eval.time_control}"]
    # -games 2 -repeat：每輪打兩局，**同一個開局位置先後手各一次**。
    # -repeat 不能省，否則白方優勢會污染 Elo 估計（規格 §3.3）。
    command += ["-games", "2", "-rounds", str(rounds), "-repeat"]

    if openings.exists():
        command += [
            "-openings",
            f"file={openings}",
            "format=pgn",
            "order=random",
        ]
    else:
        print(
            f"[提醒] 找不到開局書 {openings}，所有對局都會從起始盤面開始，\n"
            f"        確定性引擎會下出一模一樣的棋，統計沒有意義。\n"
            f"        建議先跑：python scripts/make_openings.py"
        )

    command += ["-pgnout", str(pgn_out)]
    # 每個引擎實例都會載入一份模型到 GPU，concurrency 設太高會 OOM
    command += ["-concurrency", str(cfg.eval.concurrency)]

    if sprt is not None:
        elo0, elo1, alpha, beta = sprt
        command += [
            "-sprt",
            f"elo0={elo0}",
            f"elo1={elo1}",
            f"alpha={alpha}",
            f"beta={beta}",
        ]

    return command


def engine_spec(name: str, command: str, options: dict[str, str] | None = None) -> list[str]:
    """組出單一引擎的 cutechess 參數。

    Args:
        name: 顯示名稱。
        command: 執行檔或 .bat 路徑。
        options: 要傳給引擎的 UCI option，例如 {"Skill Level": "0"}。

    Returns:
        例如 ["name=MyNet", "cmd=engine.bat"]。
    """
    spec = [f"name={name}", f"cmd={command}"]
    for key, value in (options or {}).items():
        # 含空格的選項名要整個當成一個參數傳，subprocess 用 list 就不用自己加引號
        spec.append(f"option.{key}={value}")
    return spec


def run_cutechess(command: list[str]) -> tuple[int, list[str]]:
    """執行 cutechess-cli，邊跑邊把輸出印出來。

    Args:
        command: `build_cutechess_command` 的輸出。

    Returns:
        (回傳碼, 所有輸出行)
    """
    print("執行：")
    print("  " + " ".join(command))
    print()

    lines: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        lines.append(line)
        # cutechess 每局結束會印一行 "Score of A vs B: ..."，其餘多半是雜訊
        if line.startswith(("Score of", "Finished", "Elo difference", "SPRT")):
            print("  " + line)
    process.wait()
    return process.returncode, lines


def parse_cutechess_result(lines: list[str]) -> dict:
    """解析 cutechess-cli 的輸出。

    cutechess 跑完會自己印出勝負統計與 Elo 差，直接解析最後幾行就好，
    不要自己重算（規格 §3.3）。輸出長這樣：

        Score of MyNet vs SF0: 118 - 62 - 20  [0.640] 200
        Elo difference: 99.8 +/- 34.2, LOS: 100.0 %, DrawRatio: 10.0 %
        SPRT: llr 2.96 (100.6%), lbound -2.94, ubound 2.94 - H1 was accepted

    Returns:
        {"wins", "draws", "losses", "score", "elo", "elo_error", "los",
         "sprt_status", "raw_summary", "terminations"}；解析不到的欄位不會出現。
    """
    result: dict = {}
    terminations: Counter[str] = Counter()

    for line in reversed(lines):
        stripped = line.strip()

        # 每局的結束原因：Finished game 7 (A vs B): 1-0 {White mates}
        # **一定要看這個，不要只看比分。** 比分 39-39 看起來很正常，
        # 結束原因才會讓你發現有三分之一的局數是引擎自己棄權的（見 §6.9 的 bug 紀錄）。
        if stripped.startswith("Finished game"):
            reason = re.search(r"\{([^}]+)\}", stripped)
            if reason:
                terminations[reason.group(1)] += 1

        if "elo" not in result and stripped.startswith("Elo difference:"):
            # Elo difference: 99.8 +/- 34.2, LOS: 100.0 %, DrawRatio: 10.0 %
            # 局數太少時 cutechess 會印 "+/- nan" 或 "+/- inf"，兩種都要接住
            match = re.search(
                r"Elo difference:\s*(-?[\d.]+)\s*\+/-\s*(-?[\d.]+|nan|inf)",
                stripped,
                re.IGNORECASE,
            )
            if match:
                result["elo"] = float(match.group(1))
                error = float(match.group(2))
                # nan 代表「樣本太少，算不出誤差」，統一轉成 inf 比較好懂
                result["elo_error"] = float("inf") if math.isnan(error) else error
            los = re.search(r"LOS:\s*([\d.]+)\s*%", stripped)
            if los:
                result["los"] = float(los.group(1))

        if "wins" not in result and stripped.startswith("Score of"):
            # Score of MyNet vs SF0: 118 - 62 - 20  [0.640] 200
            match = re.search(r":\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\[([\d.]+)\]\s*(\d+)", stripped)
            if match:
                result["wins"] = int(match.group(1))
                result["losses"] = int(match.group(2))
                result["draws"] = int(match.group(3))
                result["score"] = float(match.group(4))
                result["games"] = int(match.group(5))
                result["raw_summary"] = stripped

        if "sprt_status" not in result and stripped.startswith("SPRT:"):
            result["sprt_status"] = stripped
            if "H1 was accepted" in stripped:
                result["sprt_conclusion"] = "H1"     # 新版確實比較強
            elif "H0 was accepted" in stripped:
                result["sprt_conclusion"] = "H0"     # 沒有比較強

    if terminations:
        result["terminations"] = dict(terminations)
    return result


def report_terminations(parsed: dict) -> None:
    """印出結束原因分佈，並在出現棄權時大聲警告。

    「引擎回了非法著法」代表程式有 bug，不是棋力問題。這種局的勝負完全是雜訊，
    不處理的話 Elo 與 SPRT 會被嚴重污染 —— 而且比分看起來完全正常。
    """
    terminations = parsed.get("terminations")
    if not terminations:
        return

    total = sum(terminations.values())
    print("\n  結束原因：")
    for reason, count in sorted(terminations.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4} 局（{count / total * 100:>5.1f} %）{reason}")

    bad = sum(c for r, c in terminations.items() if "illegal move" in r.lower())
    if bad:
        print(
            f"\n  [警告] 有 {bad} 局（{bad / total * 100:.1f} %）是引擎走出非法著法而判負。\n"
            f"         這是程式 bug 不是棋力問題，這場的 Elo 與 SPRT 結果不能採信。\n"
            f"         先跑 python -m pytest tests/test_uci.py 找出原因再重跑。"
        )
    stalls = sum(c for r, c in terminations.items() if "stall" in r.lower() or "time" in r.lower())
    if stalls:
        print(f"\n  [注意] 有 {stalls} 局是逾時判負，時控可能太緊（目前 {cfg_time_control_hint()}）。")


def cfg_time_control_hint() -> str:
    """給警告訊息用的時控字串（避免為了印一行字把 cfg 傳進來）。"""
    return "見 config.yaml 的 eval.time_control"


def evaluate_tournament(cfg: Config, checkpoint: str, skill_level: int, rounds: int) -> dict:
    """用 cutechess-cli 讓本專案的引擎跟 Stockfish 對打。"""
    engine_path = cfg.resolve_path(cfg.eval.engine_bat)
    stockfish = cfg.resolve_path(cfg.eval.stockfish_path)
    if not stockfish.exists():
        raise SystemExit(
            f"找不到 Stockfish：{stockfish}\n"
            f"請到 https://stockfishchess.org/download/ 下載並放到 bin\\stockfish.exe"
        )

    pgn_out = PROJECT_ROOT / "logs" / f"tournament_{time.strftime('%Y%m%d_%H%M%S')}.pgn"
    command = build_cutechess_command(
        cfg,
        engine_spec("MyNet", str(engine_path), {"Checkpoint": checkpoint}),
        engine_spec(f"SF{skill_level}", str(stockfish), {"Skill Level": str(skill_level)}),
        rounds,
        pgn_out,
    )

    code, lines = run_cutechess(command)
    parsed = parse_cutechess_result(lines)
    parsed["returncode"] = code
    parsed["pgn"] = str(pgn_out)
    parsed["opponent"] = f"Stockfish Skill Level {skill_level}"

    print(f"\n{'=' * 60}")
    print(f"對 Stockfish Skill Level {skill_level}（cutechess-cli，{rounds * 2} 局）")
    print(f"{'=' * 60}")
    if "wins" in parsed:
        print(f"  勝/敗/和: {parsed['wins']} / {parsed['losses']} / {parsed['draws']}")
        print(f"  得分率  : {parsed['score'] * 100:.1f}%")
    if "elo" in parsed:
        print(f"  Elo 差  : {parsed['elo']:+.1f} +/- {parsed['elo_error']:.1f}")
    if "los" in parsed:
        print(f"  LOS     : {parsed['los']:.1f}%（對方比較強的機率的反面）")
    print(f"  對局記錄: {pgn_out}")
    report_terminations(parsed)
    return parsed


def evaluate_sprt(
    cfg: Config,
    new_checkpoint: str,
    old_checkpoint: str,
    rounds: int,
    new_options: dict[str, str] | None = None,
    old_options: dict[str, str] | None = None,
    new_name: str = "new",
    old_name: str = "old",
) -> dict:
    """用 SPRT 判斷新的 checkpoint 是不是真的比舊的強。

    SPRT 會邊打邊做序貫檢定，一旦統計上有結論就自動停止 —— 通常幾百局就夠，
    不必固定打滿。

    **沒有 SPRT 就不要相信自己的直覺**：100 局的勝率差在 ±5% 以內幾乎沒有統計
    意義，但人眼看起來會覺得「新版明顯比較強」（規格 §3.4）。
    """
    engine_path = cfg.resolve_path(cfg.eval.engine_bat)
    pgn_out = PROJECT_ROOT / "logs" / f"sprt_{time.strftime('%Y%m%d_%H%M%S')}.pgn"

    command = build_cutechess_command(
        cfg,
        engine_spec(
            new_name, str(engine_path),
            {"Checkpoint": new_checkpoint, **(new_options or {})},
        ),
        engine_spec(
            old_name, str(engine_path),
            {"Checkpoint": old_checkpoint, **(old_options or {})},
        ),
        rounds,
        pgn_out,
        sprt=(cfg.eval.sprt_elo0, cfg.eval.sprt_elo1, cfg.eval.sprt_alpha, cfg.eval.sprt_beta),
    )

    code, lines = run_cutechess(command)
    parsed = parse_cutechess_result(lines)
    parsed["returncode"] = code
    parsed["pgn"] = str(pgn_out)
    parsed["new"] = new_checkpoint
    parsed["old"] = old_checkpoint

    print(f"\n{'=' * 60}")
    # 用引擎名稱而不是 checkpoint 路徑：比較搜尋方式時兩邊 checkpoint 是同一個，
    # 印路徑會變成「best.pt vs best.pt」，看不出在比什麼
    print(f"SPRT：{new_name} vs {old_name}")
    if new_checkpoint != old_checkpoint:
        print(f"  {new_name} = {new_checkpoint}")
        print(f"  {old_name} = {old_checkpoint}")
    print(f"  H0 = 沒有比較強（elo0={cfg.eval.sprt_elo0}）")
    print(f"  H1 = 強 {cfg.eval.sprt_elo1} Elo 以上")
    print(f"{'=' * 60}")
    if "wins" in parsed:
        print(f"  勝/敗/和: {parsed['wins']} / {parsed['losses']} / {parsed['draws']}")
    if "elo" in parsed:
        print(f"  Elo 差  : {parsed['elo']:+.1f} +/- {parsed['elo_error']:.1f}")
    if "sprt_status" in parsed:
        print(f"  {parsed['sprt_status']}")
    report_terminations(parsed)

    conclusion = parsed.get("sprt_conclusion")
    if conclusion == "H1":
        print(f"\n  ✓ 新版確實比較強，可以覆蓋 models/best.pt")
    elif conclusion == "H0":
        print(f"\n  ✗ 新版沒有比較強，保留舊版")
    else:
        print(f"\n  ? 打完 {rounds * 2} 局仍無結論，可以加大 --rounds 再跑")
    return parsed


# --- 主流程 -----------------------------------------------------------------


def save_report(payload: dict) -> Path:
    """把結果存成 logs/eval_{timestamp}.json。"""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="評估模型：準確率 / 對局測試 / Elo 估計",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--mode",
        type=str,
        default="accuracy",
        choices=[
            "accuracy", "match", "baseline", "tournament",
            "sprt", "puzzles", "value-quality",
        ],
        help="評估模式（tournament / sprt 需要 cutechess-cli）",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="模型 checkpoint"
    )
    parser.add_argument("--games", type=int, default=None, help="對局數（覆寫 config）")
    parser.add_argument(
        "--skill-levels",
        type=int,
        nargs="+",
        default=None,
        help="Stockfish Skill Level（預設讀 config 的 [0, 3, 5]）。不是校準過的 Elo 刻度",
    )
    parser.add_argument(
        "--uci-elo",
        type=int,
        nargs="+",
        default=None,
        help=(
            "改用 Stockfish 官方校準的 UCI_Elo 當標尺（例如 --uci-elo 1400 1600 1800）。"
            "指定後會推估我方的絕對 Elo。這比 Skill Level 準，因為 Skill Level 是靠隨機化變弱"
        ),
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        choices=["greedy", "mcts"],
        help=(
            "puzzles 模式：在**同一批題目**上跑兩種搜尋器並做 McNemar 配對檢定。"
            "例如 --compare-with mcts 會拿 greedy 當基準跟 MCTS 比"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="greedy",
        choices=["greedy", "mcts"],
        help="--compare-with 的比較基準（預設 greedy）",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="配對比較時，每桶列出幾題「基準對而對照錯」的細節（0 = 不列）",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="accuracy 模式只跑前 N 個 batch"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="對局時的 temperature（預設 0 = argmax）"
    )
    parser.add_argument(
        "--rounds", type=int, default=None, help="tournament / sprt 要打幾輪（每輪 2 局）"
    )
    parser.add_argument(
        "--mcts",
        action="store_true",
        help="sprt：改成比較「MCTS vs greedy」（同一個 checkpoint，只差搜尋方式）",
    )
    parser.add_argument(
        "--simulations", type=int, default=None, help="sprt --mcts：MCTS 的模擬次數"
    )
    parser.add_argument("--new", type=str, default=None, help="sprt：新的 checkpoint")
    parser.add_argument("--old", type=str, default=None, help="sprt：舊的 checkpoint")
    parser.add_argument(
        "--skill-level", type=int, default=0, help="tournament：Stockfish 的 Skill Level"
    )
    parser.add_argument(
        "--value-positions",
        type=int,
        default=VALUE_QUALITY_POSITIONS,
        help="value-quality：要抽幾個盤面",
    )
    parser.add_argument(
        "--value-depth",
        type=int,
        default=VALUE_QUALITY_DEPTH,
        help="value-quality：Stockfish 搜尋深度",
    )
    parser.add_argument(
        "--puzzle-count",
        type=int,
        default=PUZZLE_DEFAULT_COUNT,
        help="puzzles：要測幾題（會平均分配到各 Rating 分桶）",
    )
    parser.add_argument(
        "--puzzle-file",
        type=str,
        default=None,
        help="puzzles：本機的 lichess_db_puzzle.csv.zst（不給就直接從網路串流）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device

    # tournament / sprt 是把對局交給 cutechess-cli 跑，本行程不需要載入模型
    # （模型是由 cutechess 啟動的 engine.bat 子行程各自載入的）
    if args.mode in ("tournament", "sprt"):
        rounds = args.rounds or cfg.eval.tournament_rounds
        payload: dict = {
            "mode": args.mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rounds": rounds,
            "time_control": cfg.eval.time_control,
        }
        if args.mode == "tournament":
            payload["result"] = evaluate_tournament(
                cfg, args.checkpoint, args.skill_level, rounds
            )
        else:
            if args.mcts:
                # MCTS vs greedy：同一個 checkpoint，只差搜尋方式（規格 §6.6）
                simulations = args.simulations or (cfg.mcts or {}).get("simulations", 800)
                payload["result"] = evaluate_sprt(
                    cfg, args.checkpoint, args.checkpoint, rounds,
                    new_options={"MCTS": "true", "Simulations": str(simulations)},
                    old_options={"MCTS": "false"},
                    new_name=f"MCTS{simulations}",
                    old_name="greedy",
                )
            else:
                new_ckpt = args.new or args.checkpoint
                old_ckpt = args.old or "models/best.pt"
                if Path(new_ckpt).resolve() == Path(old_ckpt).resolve():
                    raise SystemExit(
                        f"--new 與 --old 是同一個檔案（{new_ckpt}），比了也沒意義。\n"
                        f"用法：python -m src.evaluate --mode sprt "
                        f"--new models/epoch_12.pt --old models/best.pt\n"
                        f"或比較搜尋方式：python -m src.evaluate --mode sprt --mcts"
                    )
                payload["result"] = evaluate_sprt(cfg, new_ckpt, old_ckpt, rounds)
        path = save_report(payload)
        print(f"\n結果已存到 {path}")
        return

    device = resolve_device(cfg.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    model, checkpoint = ChessNet.from_checkpoint(checkpoint_path, device=device)

    print(f"模型      : {checkpoint_path}")
    print(f"參數量    : {model.count_parameters():,}")
    print(f"訓練 epoch: {checkpoint.get('epoch', '?')}")
    print(f"裝置      : {device}")

    payload: dict = {
        "checkpoint": str(checkpoint_path),
        "mode": args.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": checkpoint.get("epoch"),
    }

    if args.mode == "accuracy":
        metrics = evaluate_accuracy(model, cfg, device, max_batches=args.max_batches)
        print(f"\n{'=' * 60}")
        print("val 集準確率")
        print(f"{'=' * 60}")
        print(f"  盤面數      : {metrics['positions']:,}")
        print(f"  policy top-1: {metrics['policy_top1'] * 100:.2f}%")
        print(f"  policy top-5: {metrics['policy_top5'] * 100:.2f}%")
        print(f"  policy loss : {metrics['policy_loss']:.4f}")
        print(f"  value MAE   : {metrics['value_mae']:.4f}")
        if metrics["policy_top1"] < 0.10:
            print(
                "\n  ** top-1 低於 10%，八成是著法編碼或鏡射寫錯了 **\n"
                "  下一步：python -m pytest tests/test_encoding.py"
            )
        payload["metrics"] = metrics
    else:
        searcher = GreedySearcher(model, device, cfg, temperature=args.temperature)
        if args.mode == "baseline":
            payload["result"] = evaluate_baseline(searcher, cfg, args.games)
        elif args.mode == "value-quality":
            payload["result"] = evaluate_value_quality(
                model, cfg, device, args.value_positions, args.value_depth
            )
        elif args.mode == "puzzles":
            from src.search.mcts import MCTSSearcher

            puzzle_file = Path(args.puzzle_file) if args.puzzle_file else None
            simulations = args.simulations or (cfg.mcts or {}).get("simulations", 800)

            if args.compare_with:
                # 配對比較：同一批題目跑兩種搜尋器，用 McNemar 檢定
                # （獨立樣本的標準誤會嚴重高估雜訊，見 mcnemar_exact_p 的說明）
                def build(kind: str) -> tuple[Searcher, str]:
                    if kind == "greedy":
                        return GreedySearcher(model, device, cfg), "greedy"
                    return (
                        MCTSSearcher(model, device, cfg, simulations=simulations),
                        f"MCTS{simulations}",
                    )

                searcher_a, label_a = build(args.baseline)
                searcher_b, label_b = build(args.compare_with)
                payload["compare"] = {"a": label_a, "b": label_b}
                payload["result"] = evaluate_puzzles_paired(
                    searcher_a, searcher_b, label_a, label_b,
                    args.puzzle_count, puzzle_file, args.show_failures,
                )
            else:
                if args.mcts:
                    # 接上 MCTS 重跑同一批題目，用來量搜尋到底補了多少（規格 §6.6）
                    searcher = MCTSSearcher(model, device, cfg, simulations=simulations)
                    print(f"使用 MCTS（simulations={simulations}）")
                payload["mcts"] = bool(args.mcts)
                payload["result"] = evaluate_puzzles(
                    searcher, args.puzzle_count, puzzle_file
                )
        else:
            if args.uci_elo:
                # 指定了校準刻度就只用它，不要跟 Skill Level 混在同一次報告裡
                # —— 兩者不是同一個尺標，並排會讓人以為可以互相換算。
                opponents = stockfish_opponents(None, args.uci_elo)
            else:
                opponents = stockfish_opponents(
                    args.skill_levels or cfg.eval.skill_levels, None
                )
            payload["results"] = evaluate_match(searcher, cfg, opponents, args.games)

    path = save_report(payload)
    print(f"\n結果已存到 {path}")


if __name__ == "__main__":
    main()
