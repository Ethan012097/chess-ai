"""網頁棋盤的後端（規格 §4.2）。

啟動：

    python -m src.web.server                       # 最方便，會自己開 uvicorn
    uvicorn src.web.server:app --host 127.0.0.1 --port 8000

然後開瀏覽器到 http://127.0.0.1:8000（同源，不需要 CORS 設定）。

--------------------------------------------------------------------------
三個設計決定（規格 §4.1 / §4.2）
--------------------------------------------------------------------------

1. **重用既有的 `Searcher` 介面**，不另外寫一套推論邏輯。網頁看到的著法與
   `play.py`、`evaluate.py` 看到的完全一致，不會出現「網頁顯示的跟實際下的不一樣」。
2. **模型只在啟動時載入一次**存成模組層級的全域。每次請求重載的話，
   一步棋要等好幾秒，而且 VRAM 會被吃光。
3. 推論端點用**同步 `def`**（不是 `async def`）。FastAPI 看到同步函式會自動
   丟到 threadpool 執行，不會阻塞事件迴圈；寫成 `async def` 反而會把整個
   伺服器卡住（前向傳播是 CPU/GPU 密集的同步呼叫）。

--------------------------------------------------------------------------
視角約定（整個專案錯誤率最高的地方，每次出現都要講清楚）
--------------------------------------------------------------------------

- `Searcher` 與 value head 回傳的 value 一律是**當前走棋方**的視角。
- API 另外回傳 `value_white` 與 `cp`，那是**白方**視角。
- **視角轉換全部集中在後端這一支檔案裡，前端只認 `value_white`。**
  這是避免評估條畫反最有效的做法。
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, Config, add_common_args, load_config
from src.model import ChessNet, resolve_device
from src.move_info import POLICY_TOP_N, MoveInfo, value_to_cp
from src.pgn_writer import build_game
from src.search.greedy import GreedySearcher
from src.search.mcts import MCTSSearcher, terminal_value

# 前端就這一個檔（規格 §4.1：不要有建置流程）
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

DEFAULT_CHECKPOINT = "models/best.pt"
DEFAULT_TOP_K = 5
# 前端的滑桿是 1–8，跟 MoveInfo.policy_top 存的數量對齊
MAX_TOP_K = POLICY_TOP_N

# 用 uvicorn 直接啟動時（沒有命令列參數）改從環境變數讀設定
ENV_CHECKPOINT = "CHESS_AI_CHECKPOINT"
ENV_DEVICE = "CHESS_AI_DEVICE"
ENV_PRESET = "CHESS_AI_PRESET"
ENV_MCTS = "CHESS_AI_MCTS"
ENV_SIMULATIONS = "CHESS_AI_SIMULATIONS"


# --- 全域狀態 ---------------------------------------------------------------
# 這裡開 dataclass 的理由：模型、裝置、searcher、鎖這幾樣東西一定要一起傳，
# 拆成四個全域變數只會更難追。它純粹是「一包資料」，邏輯都寫在下面的函式裡。


@dataclass
class EngineState:
    """載入好的模型與搜尋器。整個 app 共用一份。

    Attributes:
        cfg: 設定。
        device: 推論裝置。
        model: 網路本體（切換 searcher 時不會重載）。
        checkpoint_path: 目前載入的 checkpoint，`/api/health` 會回報。
        epoch: checkpoint 裡記的訓練 epoch。
        use_mcts: True 表示箭頭來源是 MCTS 訪問次數，False 是 policy 機率。
        simulations: MCTS 每步的模擬次數。
        searcher: 實際下棋的搜尋器。
        lock: 推論用的鎖，見下方說明。
    """

    cfg: Config
    device: torch.device
    model: ChessNet
    checkpoint_path: Path
    epoch: Any
    use_mcts: bool
    simulations: int
    searcher: GreedySearcher | MCTSSearcher
    # FastAPI 會同時跑多個同步端點（threadpool），但 searcher 有狀態
    # （MCTS 的 last_root、臨時覆寫的 temperature），同時進來會互相蓋掉。
    # 這是單人展示用的伺服器，直接用一把鎖把推論串行化最簡單也最安全。
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATE: EngineState | None = None


def build_state(
    checkpoint: str = DEFAULT_CHECKPOINT,
    cfg: Config | None = None,
    use_mcts: bool = False,
    simulations: int | None = None,
) -> EngineState:
    """載入 checkpoint 並組出 `EngineState`。

    Args:
        checkpoint: checkpoint 路徑（相對路徑以專案根目錄為基準）。
        cfg: 設定，None 表示讀預設的 config.yaml。
        use_mcts: 是否用 MCTS 當搜尋器。
        simulations: MCTS 模擬次數，None 表示讀 cfg.mcts。

    Returns:
        EngineState。

    Raises:
        FileNotFoundError: checkpoint 不存在（訊息會告訴你要先跑哪一行）。
    """
    cfg = cfg if cfg is not None else load_config()
    device = resolve_device(cfg.device)

    path = Path(checkpoint)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    model, ckpt = ChessNet.from_checkpoint(path, device=device)

    sims = simulations if simulations is not None else int((cfg.mcts or {}).get("simulations", 800))
    state = EngineState(
        cfg=cfg,
        device=device,
        model=model,
        checkpoint_path=path,
        epoch=ckpt.get("epoch"),
        use_mcts=use_mcts,
        simulations=sims,
        searcher=GreedySearcher(model, device, cfg),
    )
    state.searcher = make_searcher(state)
    return state


def make_searcher(state: EngineState) -> GreedySearcher | MCTSSearcher:
    """依 `use_mcts` 建立搜尋器。模型是共用的，切換時不會重載。"""
    if state.use_mcts:
        return MCTSSearcher(
            state.model, state.device, state.cfg, simulations=state.simulations
        )
    return GreedySearcher(state.model, state.device, state.cfg)


def install_state(state: EngineState) -> None:
    """直接指定全域狀態（測試用，可以塞隨機權重的小模型進來）。"""
    global _STATE
    _STATE = state


def get_state() -> EngineState:
    """取得全域狀態，第一次呼叫時才載入模型。

    刻意用「第一次呼叫才載入」而不是 import 時載入：這樣測試可以先
    `install_state()` 塞一個小模型進來，不需要真的有 `models/best.pt`。
    """
    global _STATE
    if _STATE is None:
        cfg = load_config(preset=os.environ.get(ENV_PRESET) or None)
        if os.environ.get(ENV_DEVICE):
            cfg.device = os.environ[ENV_DEVICE]
        _STATE = build_state(
            checkpoint=os.environ.get(ENV_CHECKPOINT, DEFAULT_CHECKPOINT),
            cfg=cfg,
            use_mcts=os.environ.get(ENV_MCTS, "0") == "1",
            simulations=int(os.environ[ENV_SIMULATIONS])
            if os.environ.get(ENV_SIMULATIONS)
            else None,
        )
    return _STATE


# --- 請求 / 回應的資料結構 ---------------------------------------------------


class AnalyseRequest(BaseModel):
    """`POST /api/analyse` 的輸入。"""

    fen: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    # 自動播放時給一點隨機性，不然每一局都會下得一模一樣。0 = 完全照最高分走。
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class MoveRequest(BaseModel):
    """`POST /api/move` 的輸入。"""

    fen: str
    uci: str


class PgnMove(BaseModel):
    """`POST /api/pgn` 裡的一步棋（前端把 analyse 的結果原封不動存起來再送回來）。"""

    uci: str
    value: float = 0.0
    policy_top: list[tuple[str, float]] = Field(default_factory=list)
    visits: dict[str, int] | None = None
    elapsed_ms: int = 0


class PgnRequest(BaseModel):
    """`POST /api/pgn` 的輸入。"""

    moves: list[PgnMove]
    headers: dict[str, str] = Field(default_factory=dict)


class ConfigRequest(BaseModel):
    """`POST /api/config` 的輸入：在網頁上切換搜尋方式。"""

    mcts: bool | None = None
    simulations: int | None = Field(default=None, ge=1, le=100_000)


# --- 共用小工具 -------------------------------------------------------------


def parse_fen(fen: str) -> chess.Board:
    """把 FEN 字串轉成盤面，格式錯誤時回 400 而不是 500。

    Args:
        fen: FEN 字串。

    Returns:
        chess.Board。

    Raises:
        HTTPException: FEN 不合法。
    """
    try:
        return chess.Board(fen.strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"FEN 不合法：{exc}。標準起始盤面是 {chess.STARTING_FEN}",
        ) from exc


def game_over_info(board: chess.Board) -> tuple[bool, str | None, str | None]:
    """判斷對局是否結束。

    Args:
        board: 盤面。

    Returns:
        (is_game_over, result, reason)
          result 是 "1-0" / "0-1" / "1/2-1/2"，未結束時為 None。
          reason 是給人看的中文說明。

    注意：從 FEN 重建的盤面沒有歷史，所以偵測不到三次重複。
    但五十步計數在 FEN 裡，七十五步規則仍然會生效，自動播放不會無限跑下去。
    """
    if not board.is_game_over(claim_draw=True):
        return False, None, None

    if board.is_checkmate():
        reason = "將死"
    elif board.is_stalemate():
        reason = "逼和（無子可動）"
    elif board.is_insufficient_material():
        reason = "子力不足"
    elif board.is_seventyfive_moves() or board.can_claim_fifty_moves():
        reason = "五十步規則"
    elif board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
        reason = "重複盤面"
    else:
        reason = "和局"
    return True, board.result(claim_draw=True), reason


def value_payload(value: float, turn: chess.Color) -> dict[str, float]:
    """把走棋方視角的 value 轉成前端要的白方視角欄位。

    **整個專案的視角轉換就集中在這裡。** 前端拿到 `value_white` 直接畫，
    不做任何加減號。

    Args:
        value: **當前走棋方**視角的 value，-1 ~ 1。
        turn: 當前走棋方。

    Returns:
        {"value": 走棋方視角, "value_white": 白方視角, "cp": 白方視角的百分兵值}
    """
    value_white = value if turn == chess.WHITE else -value
    return {
        "value": round(float(value), 4),
        "value_white": round(float(value_white), 4),
        "cp": value_to_cp(value_white),
    }


# --- 推論 -------------------------------------------------------------------


def analyse_position(
    state: EngineState, board: chess.Board, top_k: int, temperature: float = 0.0
) -> dict[str, Any]:
    """分析一個盤面：value、候選著法機率、以及 AI 實際會走的那一步。

    箭頭的資料來源有兩種（規格 §4.4），**兩者意義不同不能混為一談**：
      - greedy：policy head 的機率，代表「直覺想走哪裡」
      - MCTS：訪問次數比例，代表「想過之後認為哪裡值得」

    Args:
        state: 全域狀態。
        board: 要分析的盤面。
        top_k: 回傳前幾名候選著法。
        temperature: 0 = 取最高分；>0 = 依機率抽樣（自動播放製造變化用）。

    Returns:
        可以直接丟給 FastAPI 序列化的 dict，欄位見 §4.2。
        `value` 是走棋方視角，`value_white` / `cp` 是白方視角。
    """
    start = time.perf_counter()
    is_over, result, reason = game_over_info(board)

    if is_over:
        # 終局不呼叫網路：分數是確定的，用網路去猜反而會給出矛盾的數字。
        # terminal_value 回傳的也是**當前走棋方**視角（被將死 = -1）。
        final = terminal_value(board)
        payload = value_payload(final if final is not None else 0.0, board.turn)
        payload.update(
            moves=[],
            best=None,
            is_game_over=True,
            result=result,
            result_reason=reason,
            source="mcts" if state.use_mcts else "policy",
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
        return payload

    with state.lock:
        if state.use_mcts:
            value, probs, visits, best = _analyse_mcts(state, board, temperature)
        else:
            value, probs, visits, best = _analyse_policy(state, board, temperature)

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    moves = [
        {
            "uci": move.uci(),
            "san": board.san(move),
            "prob": round(float(prob), 4),
            "visits": visits.get(move.uci()) if visits else None,
        }
        for move, prob in ranked
    ]

    payload = value_payload(value, board.turn)
    payload.update(
        moves=moves,
        best={"uci": best.uci(), "san": board.san(best)},
        is_game_over=False,
        result=None,
        result_reason=None,
        source="mcts" if state.use_mcts else "policy",
        elapsed_ms=int((time.perf_counter() - start) * 1000),
    )
    return payload


def _analyse_policy(
    state: EngineState, board: chess.Board, temperature: float
) -> tuple[float, dict[chess.Move, float], dict[str, int] | None, chess.Move]:
    """greedy 版本的分析。

    Returns:
        (value, {著法: policy 機率}, None, AI 實際會走的著法)
        value 是**當前走棋方**視角。

    這裡會做兩次前向傳播（一次拿箭頭、一次讓 `select_move` 跑將死檢查與送子檢查）。
    多花的幾毫秒換來「畫出來的箭頭」與「真的會走的棋」保證同源，很划算。
    箭頭最粗的那條有時不等於實際走的那步 —— 那正是送子檢查發揮作用的時候。
    """
    searcher = state.searcher
    assert isinstance(searcher, GreedySearcher)
    value, probs = searcher.analyse(board)

    original = searcher.temperature
    searcher.temperature = temperature
    try:
        best = searcher.select_move(board)
    finally:
        searcher.temperature = original
    return value, probs, None, best


def _analyse_mcts(
    state: EngineState, board: chess.Board, temperature: float
) -> tuple[float, dict[chess.Move, float], dict[str, int], chess.Move]:
    """MCTS 版本的分析。

    Returns:
        (value, {著法: 訪問次數比例}, {uci: 訪問次數}, AI 實際會走的著法)
        value 取自根節點的 Q，也是**當前走棋方**視角
        （`_backup` 每往上一層取一次負號，所以根節點的 Q 就是根節點走棋方的視角）。

    只跑一次搜尋。若改呼叫 `searcher.move_probabilities()` 會再搜一次，白白慢一倍。
    """
    searcher = state.searcher
    assert isinstance(searcher, MCTSSearcher)
    root = searcher.run_simulations(board)
    visit_counts = searcher.visit_counts(board)

    total = sum(visit_counts.values())
    if total == 0:
        # 一次模擬都沒跑成（理論上不會發生），退回均勻分佈
        legal = list(board.legal_moves)
        uniform = {m: 1.0 / len(legal) for m in legal}
        return root.q(), uniform, visit_counts, legal[0]

    probs = {
        chess.Move.from_uci(uci): count / total for uci, count in visit_counts.items()
    }
    if temperature > 0:
        best = _sample_by_temperature(probs, temperature, searcher.rng)
    else:
        best = max(probs, key=probs.get)
    return root.q(), probs, visit_counts, best


def _sample_by_temperature(
    probs: dict[chess.Move, float], temperature: float, rng: np.random.Generator
) -> chess.Move:
    """依 p^(1/τ) 抽樣一個著法（自動播放用，避免每局都下得一模一樣）。

    Args:
        probs: {著法: 機率}。
        temperature: >0。越大越平均。
        rng: 亂數產生器（沿用 searcher 的，才吃得到 config 的 seed）。

    Returns:
        抽中的著法。
    """
    moves = list(probs)
    weights = np.array([probs[m] for m in moves], dtype=np.float64) ** (1.0 / temperature)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return max(probs, key=probs.get)
    weights /= total
    return moves[int(rng.choice(len(moves), p=weights))]


# --- FastAPI app ------------------------------------------------------------

app = FastAPI(title="chess-ai web board", version="1.0")


@app.get("/")
def index() -> FileResponse:
    """回傳前端頁面（規格 §4.2）。"""
    if not INDEX_HTML.exists():
        raise HTTPException(
            status_code=500,
            detail=f"找不到 {INDEX_HTML}。前端只有這一個檔，請確認它沒有被刪掉。",
        )
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """回報目前載入的是哪個 checkpoint，方便比較不同版本（規格 §4.2）。"""
    state = get_state()
    return {
        "model_path": str(state.checkpoint_path),
        "device": str(state.device),
        "params": state.model.count_parameters(),
        "epoch": state.epoch,
        "preset": state.cfg.preset,
        "channels": state.cfg.model.channels,
        "blocks": state.cfg.model.blocks,
        "searcher": "mcts" if state.use_mcts else "policy",
        "simulations": state.simulations if state.use_mcts else None,
    }


@app.post("/api/analyse")
def analyse(req: AnalyseRequest) -> dict[str, Any]:
    """分析盤面，回傳評估與候選著法（同步 def，見檔頭說明）。"""
    state = get_state()
    board = parse_fen(req.fen)
    return analyse_position(state, board, req.top_k, req.temperature)


@app.post("/api/move")
def move(req: MoveRequest) -> dict[str, Any]:
    """在盤面上走一步，回傳新的 FEN。

    合法性由 python-chess 判斷（前端的 chess.js 只是為了拖曳時能即時回饋）。
    非法著法回傳 `legal: false` 與**原本的 FEN**，前端據此把棋子彈回去，
    盤面不會進入錯誤狀態。
    """
    board = parse_fen(req.fen)
    try:
        candidate = chess.Move.from_uci(req.uci.strip())
    except ValueError:
        return {"legal": False, "fen": board.fen(), "reason": f"看不懂的著法：{req.uci}"}

    if candidate not in board.legal_moves:
        return {"legal": False, "fen": board.fen(), "reason": "這步不合法"}

    san = board.san(candidate)
    board.push(candidate)
    is_over, result, reason = game_over_info(board)
    return {
        "legal": True,
        "fen": board.fen(),
        "san": san,
        "uci": candidate.uci(),
        "is_game_over": is_over,
        "result": result,
        "result_reason": reason,
        "is_check": board.is_check(),
    }


@app.post("/api/pgn")
def export_pgn(req: PgnRequest) -> dict[str, str]:
    """把整局棋輸出成 PGN（含 lichess 認得的 `[%eval]` 註解）。

    刻意在後端做而不是用前端的 chess.js `.pgn()`：這樣可以重用 `pgn_writer.py`，
    輸出跟 CLI 對弈、自我對弈完全同一個格式，貼到 lichess 會畫出評分曲線。
    """
    state = get_state()
    board = chess.Board()
    infos: list[MoveInfo] = []

    for i, item in enumerate(req.moves):
        try:
            mv = chess.Move.from_uci(item.uci.strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"第 {i + 1} 步著法格式錯誤：{item.uci}") from exc
        if mv not in board.legal_moves:
            raise HTTPException(
                status_code=400,
                detail=f"第 {i + 1} 步 {item.uci} 在該盤面不合法，PGN 無法組成。",
            )
        infos.append(
            MoveInfo(
                move=mv,
                san=board.san(mv),
                value=item.value,
                policy_top=[(u, float(p)) for u, p in item.policy_top],
                visits=item.visits,
                elapsed_ms=item.elapsed_ms,
                fen_before=board.fen(),
            )
        )
        board.push(mv)

    headers = {
        "Event": "web board",
        "Site": "http://127.0.0.1:8000",
        "ModelCheckpoint": str(state.checkpoint_path),
        **req.headers,
    }
    return {"pgn": str(build_game(board, infos, headers))}


@app.post("/api/config")
def set_config(req: ConfigRequest) -> dict[str, Any]:
    """在網頁上切換 policy / MCTS（模型不會重載，只換搜尋器）。"""
    state = get_state()
    with state.lock:
        if req.simulations is not None:
            state.simulations = req.simulations
        if req.mcts is not None:
            state.use_mcts = req.mcts
        state.searcher = make_searcher(state)
    return health()


# --- 命令列 -----------------------------------------------------------------


def main() -> None:
    """`python -m src.web.server` 直接啟動伺服器。"""
    parser = argparse.ArgumentParser(description="網頁棋盤（策略箭頭 + 評估條）")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mcts", action="store_true", help="用 MCTS 而不是 policy（箭頭會變成訪問次數比例）")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS 每步的模擬次數")
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device

    install_state(
        build_state(
            checkpoint=args.checkpoint,
            cfg=cfg,
            use_mcts=args.mcts,
            simulations=args.simulations,
        )
    )
    state = get_state()
    print(f"模型   : {state.checkpoint_path}（{state.model.count_parameters():,} 參數）")
    print(f"裝置   : {state.device}")
    print(f"搜尋   : {'MCTS ' + str(state.simulations) + ' 次模擬' if state.use_mcts else 'policy（greedy）'}")
    print(f"\n開瀏覽器到  http://{args.host}:{args.port}  （Ctrl+C 停止）\n")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
