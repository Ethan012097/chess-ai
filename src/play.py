"""跟 AI 下棋。兩種模式：

    python -m src.play --mode cli          # 終端機文字棋盤，人類與 AI 對弈
    python -m src.play --mode uci          # UCI 協定，可掛進 Cute Chess / Arena

CLI 模式的指令：
    e4 / e2e4   走一步（SAN 或 UCI 都可以）
    undo        悔棋（一次退兩步，人與 AI 各一步）
    hint        看 AI 覺得最好的前 5 步
    fen         印出目前盤面的 FEN
    save        把這局存成 PGN
    flip        換邊（讓 AI 先走）
    quit        離開
"""

from __future__ import annotations

# UCI 的坑之二：任何跑到 stdout 的東西都會讓 GUI 解析失敗，PyTorch 的
# UserWarning 也不例外。所以在 import torch 之前就把警告關掉。
# （play.py 只負責下棋，不做訓練，關掉警告沒有損失。）
import warnings

warnings.filterwarnings("ignore")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import chess  # noqa: E402
import torch  # noqa: E402

from src.config import PROJECT_ROOT, Config, add_common_args, load_config  # noqa: E402
from src.model import ChessNet, resolve_device  # noqa: E402
from src.move_info import MoveInfo, make_move_info, value_to_cp  # noqa: E402
from src.pgn_writer import write_game  # noqa: E402
from src.encoding import legal_indices  # noqa: E402
from src.search.greedy import GreedySearcher  # noqa: E402
from src.search.mcts import (  # noqa: E402
    TIME_SAFETY_MARGIN_MS,
    MCTSSearcher,
    allocate_time_ms,
)

DEFAULT_CHECKPOINT = "models/best.pt"
HINT_COUNT = 5
UCI_ENGINE_NAME = "ChessAI-SL"
UCI_ENGINE_AUTHOR = "chess-ai (Phase 1, supervised)"
# UCI 的 spin 選項只能是整數，所以溫度用「百分比」表示：30 代表 temperature=0.3
TEMPERATURE_SPIN_SCALE = 100.0

# 終端機用的棋子符號。用 Unicode 圖形比字母好認很多。
PIECE_SYMBOLS: dict[str, str] = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def load_searcher(
    checkpoint: str, cfg: Config, temperature: float | None = None
) -> tuple[GreedySearcher, ChessNet]:
    """載入模型並包成 searcher。

    Args:
        checkpoint: checkpoint 路徑。
        cfg: 設定。
        temperature: 覆寫 cfg.search.temperature。

    Returns:
        (searcher, model)
    """
    device = resolve_device(cfg.device)
    path = Path(checkpoint)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    model, _ = ChessNet.from_checkpoint(path, device=device)
    searcher = GreedySearcher(model, device, cfg, temperature=temperature)
    return searcher, model


def render_board(board: chess.Board, flipped: bool = False) -> str:
    """把盤面畫成文字棋盤（格式見 CLAUDE-PHASE2.md §1.4）。

    上下都標 file、左右都標 rank，這樣不用把視線拉到角落就能定位。

    Args:
        board: 盤面。
        flipped: True 表示從黑方視角看（黑方在下）。

    Returns:
        可以直接 print 的多行字串。
    """
    ranks = range(8) if flipped else range(7, -1, -1)
    files = range(7, -1, -1) if flipped else range(8)
    file_labels = " ".join("hgfedcba" if flipped else "abcdefgh")

    lines = [f"  {file_labels}"]
    for rank in ranks:
        cells = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            cells.append(PIECE_SYMBOLS[piece.symbol()] if piece else "·")
        lines.append(f"{rank + 1} " + " ".join(cells) + f"  {rank + 1}")
    lines.append(f"  {file_labels}")
    return "\n".join(lines)


def format_legal_moves(board: chess.Board) -> str:
    """列出所有合法著法的 SAN。

    輸入非法著法時，只說「非法」對使用者毫無幫助——直接把能走的都列出來。

    Args:
        board: 盤面。

    Returns:
        以空白分隔、每行 8 個的 SAN 列表。
    """
    sans = sorted(board.san(m) for m in board.legal_moves)
    lines = [" ".join(sans[i : i + 8]) for i in range(0, len(sans), 8)]
    return "\n".join("  " + line for line in lines)


def describe_game_over(board: chess.Board) -> str:
    """把結束原因講成人話。"""
    if board.is_checkmate():
        winner = "黑方" if board.turn == chess.WHITE else "白方"
        return f"將死！{winner}獲勝。"
    if board.is_stalemate():
        return "逼和（stalemate），和局。"
    if board.is_insufficient_material():
        return "子力不足，和局。"
    if board.is_fifty_moves():
        return "五十步規則，和局。"
    if board.is_repetition():
        return "三次重複，和局。"
    return f"棋局結束：{board.result()}"


def parse_user_move(board: chess.Board, text: str) -> chess.Move | None:
    """把使用者輸入的字串解析成著法，SAN 與 UCI 都接受。

    **先試 SAN 再試 UCI**（規格 §1.4）：因為 SAN 才是人類習慣寫的（`e4`、`Nf3`），
    而且有些字串兩種都解得動，優先採用人類的寫法比較不會意外。

    Args:
        board: 目前盤面。
        text: 使用者輸入，例如 `e4`、`Nf3`、`e2e4`、`O-O`。

    Returns:
        合法著法；解析失敗或不合法時回傳 None。
    """
    text = text.strip()
    if not text:
        return None
    try:
        return board.parse_san(text)
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def print_hints(searcher: GreedySearcher, board: chess.Board) -> None:
    """印出 AI 認為最好的前幾步。"""
    probs = searcher.move_probabilities(board)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:HINT_COUNT]
    print(f"\nAI 的前 {len(ranked)} 個選擇：")
    for move, p in ranked:
        print(f"  {board.san(move):8s} ({move.uci()})  {p * 100:5.1f}%")


def save_current_game(
    board: chess.Board,
    move_infos: list[MoveInfo],
    checkpoint: str,
    human_is_white: bool,
    path: Path | None = None,
) -> Path:
    """把目前這局存成 PGN（CLI 的 `save` 指令）。

    Args:
        board: 目前盤面。
        move_infos: 每一步的紀錄（人類走的那幾步沒有網路評估，value 記 0）。
        checkpoint: 模型檔名，寫進 ModelCheckpoint 標頭。
        human_is_white: 人類執白與否，用來填 White / Black 標頭。
        path: 輸出路徑，None 表示自動用時間戳。

    Returns:
        實際寫出的路徑。
    """
    if path is None:
        path = PROJECT_ROOT / "logs" / "games" / f"cli_{time.strftime('%Y%m%d_%H%M%S')}.pgn"
    write_game(
        board,
        move_infos,
        {
            "Event": "CLI human vs AI",
            "White": "Human" if human_is_white else "MyNet",
            "Black": "MyNet" if human_is_white else "Human",
            "Result": board.result(claim_draw=True),
            "ModelCheckpoint": checkpoint,
        },
        path,
    )
    return path


def play_cli(
    searcher: GreedySearcher,
    cfg: Config,
    human_is_white: bool,
    checkpoint: str = "?",
) -> None:
    """終端機對弈模式。

    Args:
        searcher: AI。
        cfg: 設定。
        human_is_white: 人類是否執白（執白先走）。
        checkpoint: 模型檔名，存 PGN 時寫進標頭。
    """
    board = chess.Board()
    move_infos: list[MoveInfo] = []

    print("=" * 62)
    print("人機對弈。著法可以用 SAN（e4、Nf3、O-O）或 UCI（e2e4）。")
    print("指令：undo（悔棋兩步）  hint（提示）  fen（印出 FEN）")
    print("      save（存成 PGN）  flip（換邊）  quit（離開）")
    print(f"你執{'白' if human_is_white else '黑'}方。")
    print("=" * 62)

    while not board.is_game_over(claim_draw=True):
        human_turn = board.turn == (chess.WHITE if human_is_white else chess.BLACK)
        print()
        print(render_board(board, flipped=not human_is_white))
        print(
            f"\n第 {board.fullmove_number} 手，輪到{'你' if human_turn else 'AI'}"
            f"（{'白' if board.turn == chess.WHITE else '黑'}方）"
            f"{'  ** 將軍！**' if board.is_check() else ''}"
        )

        if human_turn:
            try:
                # lstrip 掉 BOM，理由同 play_uci
                text = input("\n你的著法 > ").lstrip("﻿").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n離開。")
                return

            if text in ("quit", "q", "exit"):
                print("離開。")
                return
            if text == "undo":
                # 悔兩步（人與 AI 各一步），這樣回到的還是輪到人類的狀態
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    move_infos = move_infos[:-2]
                    print("已悔棋一回合。")
                else:
                    print("沒有可以悔的棋。")
                continue
            if text == "hint":
                print_hints(searcher, board)
                continue
            if text == "fen":
                print(f"  {board.fen()}")
                continue
            if text == "save":
                path = save_current_game(board, move_infos, checkpoint, human_is_white)
                print(f"  已存到 {path}")
                print(f"  可以貼到 https://lichess.org/paste 看評分曲線")
                continue
            if text == "flip":
                human_is_white = not human_is_white
                print(f"換邊，你現在執{'白' if human_is_white else '黑'}方。")
                continue
            if not text:
                continue

            move = parse_user_move(board, text)
            if move is None:
                print(f"\n「{text}」不是合法著法。目前可走的有：")
                print(format_legal_moves(board))
                continue

            # 人類走的步也記下來，這樣存出來的 PGN 才是完整一局。
            # 這裡照樣跑一次網路評估：如果人類這幾步留白（value 記 0），
            # lichess 的評分曲線會每隔一步掉回 0，變成鋸齒狀的假訊號。
            # 多一次前向傳播只要幾毫秒，換到一條有意義的曲線很划算。
            human_value, human_probs = searcher.analyse(board)
            move_infos.append(
                make_move_info(board, move, human_value, human_probs, elapsed_ms=0)
            )
            print(f"你走 {board.san(move)}")
            board.push(move)
        else:
            start = time.perf_counter()
            # analyse 一次前向就同時拿到 value 與 policy，不用算兩遍
            value, probs = searcher.analyse(board)
            move = searcher.select_move(board)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            info = make_move_info(board, move, value, probs, elapsed_ms)
            move_infos.append(info)

            confidence = dict(
                (chess.Move.from_uci(u), p) for u, p in info.policy_top
            ).get(move, 0.0)
            san = board.san(move)
            board.push(move)

            # 這裡的「評估」是 AI 走這步之前、從 AI 自己視角看的分數
            print(
                f"\nAI: {san}   (信心 {confidence * 100:.0f}%, "
                f"評估 {value:+.2f}, {elapsed_ms / 1000:.2f}s)"
            )
            candidates = " | ".join(
                f"{chess.Board(info.fen_before).san(chess.Move.from_uci(u))} {p * 100:.0f}%"
                for u, p in info.policy_top[:5]
            )
            print(f"候選: {candidates}")

    print()
    print(render_board(board, flipped=not human_is_white))
    print(f"\n{describe_game_over(board)}")
    path = save_current_game(board, move_infos, checkpoint, human_is_white)
    print(f"棋譜已存到 {path}")


def log_stderr(message: str) -> None:
    """把除錯訊息印到 stderr。

    UCI 的坑之二：**stdout 只能有協定訊息**。任何除錯輸出跑到 stdout 都會讓 GUI
    解析失敗，而且失敗時 GUI 通常只會說「引擎沒回應」，不會告訴你原因。
    """
    print(message, file=sys.stderr, flush=True)


class UciEngine:
    """UCI 協定的狀態機。

    這裡開 class 的理由：UCI 是有狀態的協定 —— 盤面、各項 option、載入好的模型
    都要在指令之間保存。用函式的話每個 handler 都要把這一大包傳來傳去，
    而且 `setoption` 還要能改到它們。

    **四個必踩的坑**（規格 §2.3），程式碼中都標了對應註解：
      一、stdout 緩衝 → `run()` 開頭 reconfigure(line_buffering=True)
      二、stdout 只能有 UCI 訊息 → 除錯一律走 `log_stderr`
      三、絕不吐出非法著法 → `_choose_move` 一定檢查合法性並包 try/except
      四、升變與易位格式 → 全部交給 python-chess 的 `Move.uci()` / `push_uci()`
    """

    def __init__(self, cfg: Config, checkpoint: str, device: torch.device) -> None:
        """
        Args:
            cfg: 設定。
            checkpoint: 初始 checkpoint 路徑。
            device: 推論裝置。
        """
        self.cfg = cfg
        self.device = device
        self.board = chess.Board()
        self.checkpoint = checkpoint
        self.use_mcts = False
        self.simulations = int((cfg.mcts or {}).get("simulations", 800))
        # 模型只在這裡載入一次。每次 go 都重載的話，GUI 會覺得引擎慢得莫名其妙。
        self.model = self._load_model(checkpoint)
        self.searcher = self._make_searcher()

    # --- 模型 ---------------------------------------------------------------

    def _load_model(self, checkpoint: str) -> ChessNet:
        """載入 checkpoint。"""
        path = Path(checkpoint)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        model, _ = ChessNet.from_checkpoint(path, device=self.device)
        log_stderr(f"[uci] 已載入 {path}（{model.count_parameters():,} 參數，{self.device}）")
        return model

    def _make_searcher(self) -> GreedySearcher | MCTSSearcher:
        """依 MCTS 選項建立對應的 searcher（模型是共用的，不會重載）。"""
        if self.use_mcts:
            log_stderr(f"[uci] 使用 MCTS，simulations={self.simulations}")
            return MCTSSearcher(
                self.model, self.device, self.cfg, simulations=self.simulations
            )
        log_stderr("[uci] 使用 greedy searcher")
        return GreedySearcher(self.model, self.device, self.cfg)

    # --- 各指令的處理 --------------------------------------------------------

    def handle_uci(self) -> None:
        """握手：回報名稱、作者、可設定的 option，最後 uciok。"""
        print(f"id name {UCI_ENGINE_NAME}")
        print(f"id author {UCI_ENGINE_AUTHOR}")
        # spin 只能是整數，所以 Temperature 用百分比（30 = 0.3）
        print(
            "option name Temperature type spin default "
            f"{int(self.cfg.search.temperature * TEMPERATURE_SPIN_SCALE)} min 0 max 100"
        )
        print(f"option name TopK type spin default {self.cfg.search.top_k} min 1 max 20")
        print(f"option name Checkpoint type string default {self.checkpoint}")
        # 以下兩個是 Phase 2 預留的，目前設了也不會有作用
        print("option name MCTS type check default false")
        print(
            "option name Simulations type spin default "
            f"{self.cfg.mcts.get('simulations', 800)} min 1 max 100000"
        )
        print("uciok", flush=True)

    def handle_setoption(self, parts: list[str]) -> None:
        """處理 `setoption name <X> value <Y>`。

        選項名可能含空白（例如 Stockfish 的 "Skill Level"），所以要用 name 與
        value 兩個關鍵字切開，不能直接取固定位置。
        """
        if "name" not in parts:
            return
        name_idx = parts.index("name") + 1
        if "value" in parts:
            value_idx = parts.index("value")
            name = " ".join(parts[name_idx:value_idx])
            value = " ".join(parts[value_idx + 1 :])
        else:
            name = " ".join(parts[name_idx:])
            value = ""

        key = name.lower()
        try:
            if key == "temperature":
                self.searcher.temperature = int(value) / TEMPERATURE_SPIN_SCALE
                log_stderr(f"[uci] Temperature = {self.searcher.temperature}")
            elif key == "topk":
                self.searcher.top_k = int(value)
                log_stderr(f"[uci] TopK = {self.searcher.top_k}")
            elif key == "checkpoint":
                self.checkpoint = value
                self.model = self._load_model(value)
                self.searcher = self._make_searcher()
            elif key == "mcts":
                self.use_mcts = value.lower() == "true"
                self.searcher = self._make_searcher()
            elif key == "simulations":
                self.simulations = int(value)
                if isinstance(self.searcher, MCTSSearcher):
                    self.searcher.simulations = self.simulations
                log_stderr(f"[uci] Simulations = {self.simulations}")
            else:
                log_stderr(f"[uci] 未知的 option：{name}")
        except (ValueError, FileNotFoundError) as exc:
            # 設定失敗不可以讓引擎死掉，GUI 會直接判當機
            log_stderr(f"[uci] setoption {name} 失敗：{exc}")

    def handle_position(self, parts: list[str]) -> None:
        """處理 `position startpos|fen ... [moves ...]`。

        **每次都從頭重建 Board 再逐一 push**（規格 §2.2）。成本可以忽略，
        但比嘗試增量更新的錯誤率低太多了。
        """
        try:
            if "startpos" in parts:
                self.board = chess.Board()
                cursor = parts.index("startpos") + 1
            elif "fen" in parts:
                fen_start = parts.index("fen") + 1
                fen_end = parts.index("moves") if "moves" in parts else len(parts)
                self.board = chess.Board(" ".join(parts[fen_start:fen_end]))
                cursor = fen_end
            else:
                log_stderr(f"[uci] position 指令看不懂：{' '.join(parts)}")
                return

            if cursor < len(parts) and parts[cursor] == "moves":
                for uci in parts[cursor + 1 :]:
                    # push_uci 已經處理好升變（e7e8q）與易位（e1g1）的格式
                    self.board.push_uci(uci)
        except ValueError as exc:
            log_stderr(f"[uci] position 解析失敗：{exc}；維持原本盤面")

    @staticmethod
    def parse_go(parts: list[str]) -> dict[str, int | bool]:
        """解析 `go` 的參數。

        Phase 1 的 greedy searcher 幾乎瞬間回，這些值目前只記錄不使用；
        Phase 2 接上 MCTS 之後才需要真的做時間管理（規格 §6.5）。

        Returns:
            例如 {"movetime": 100} 或 {"wtime": 60000, "winc": 1000}。
        """
        options: dict[str, int | bool] = {}
        int_keys = {
            "movetime", "wtime", "btime", "winc", "binc",
            "depth", "nodes", "movestogo",
        }
        i = 1
        while i < len(parts):
            token = parts[i]
            if token == "infinite":
                options["infinite"] = True
            elif token == "ponder":
                options["ponder"] = True
            elif token in int_keys and i + 1 < len(parts):
                try:
                    options[token] = int(parts[i + 1])
                except ValueError:
                    pass
                i += 1
            i += 1
        return options

    def _choose_move(self) -> chess.Move:
        """選一步棋，並保證回傳的一定是合法著法。

        **坑之三：絕對不能吐出非法著法。** GUI 收到非法著法會直接判負，而且不會
        告訴你原因。所以這裡包一層 try/except：任何例外都退回「隨便走一步合法
        著法」，例外訊息印到 stderr —— 絕不崩潰、絕不沉默。

        Returns:
            保證 `move in board.legal_moves` 的著法。
        """
        legal = list(self.board.legal_moves)
        if not legal:
            return chess.Move.null()

        try:
            move = self.searcher.select_move(self.board)
            if move in self.board.legal_moves:
                return move
            log_stderr(f"[uci] 搜尋回傳非法著法 {move.uci()}，改走第一個合法著法")
        except Exception as exc:  # noqa: BLE001 - 這裡就是要攔下所有例外
            log_stderr(f"[uci] 搜尋發生例外：{type(exc).__name__}: {exc}")
        return legal[0]

    def _deadline_from_go(self, options: dict[str, int | bool]) -> float | None:
        """依 `go` 的時間參數算出這一手的截止時刻。

        greedy 幾乎瞬間回，用不到；MCTS 才需要真的做時間管理（規格 §6.5）。

        Returns:
            `time.perf_counter()` 的截止時刻；沒有時間限制時回傳 None。
        """
        if options.get("infinite"):
            return None

        movetime = options.get("movetime")
        if isinstance(movetime, int):
            budget_ms = max(movetime - TIME_SAFETY_MARGIN_MS, 1)
        else:
            key_time = "wtime" if self.board.turn == chess.WHITE else "btime"
            key_inc = "winc" if self.board.turn == chess.WHITE else "binc"
            remaining = options.get(key_time)
            if not isinstance(remaining, int):
                return None
            increment = options.get(key_inc)
            budget_ms = allocate_time_ms(
                remaining, increment if isinstance(increment, int) else 0
            )
        return time.perf_counter() + budget_ms / 1000.0

    def handle_go(self, parts: list[str]) -> None:
        """思考並輸出 `info` 數行 + `bestmove`。"""
        options = self.parse_go(parts)

        # **只有真的沒有合法著法時**才回 0000（UCI 的空著法）。
        #
        # 這裡曾經寫成 `is_game_over(claim_draw=True)`，那是個會默默輸棋的 bug：
        # 三次重複與五十步規則是「可以宣告」而不是「自動成立」，盤面上還有合法著法。
        # 我們回了 0000，Cute Chess 判定為 illegal move 直接判負 ——
        # 實測一場 80 局的對打裡有 28 局是這樣沒的。和棋要交給 GUI 去裁決，
        # 引擎該做的是繼續走棋。
        if not any(self.board.legal_moves):
            log_stderr("[uci] 沒有合法著法，回傳 bestmove 0000")
            print("bestmove 0000", flush=True)
            return

        start = time.perf_counter()
        value = 0.0
        nodes = 0
        pv: list[chess.Move] = []

        try:
            if isinstance(self.searcher, MCTSSearcher):
                deadline = self._deadline_from_go(options)
                root = self.searcher.run_simulations(self.board, deadline=deadline)
                # root.q() 是從根節點走棋方的視角，跟 score cp 的約定一致
                value = root.q()
                nodes = sum(c.visit_count for c in root.children.values())
                pv = self.searcher.principal_variation(self.board)
                move = self._pick_from_visits(root)
            else:
                # analyse 一次前向就同時取得 value 與 policy
                value, probs = self.searcher.analyse(self.board)
                nodes = len(probs)
                move = self._choose_move()
                pv = [move] if move != chess.Move.null() else []
        except Exception as exc:  # noqa: BLE001
            log_stderr(f"[uci] 搜尋失敗：{type(exc).__name__}: {exc}")
            move = self._choose_move()
            pv = [move] if move != chess.Move.null() else []

        elapsed_ms = max(int((time.perf_counter() - start) * 1000), 1)
        self._print_info(move, value, nodes, pv, elapsed_ms)
        print(f"bestmove {move.uci()}", flush=True)

    def _pick_from_visits(self, root) -> chess.Move:
        """從 MCTS 的根節點挑訪問次數最多的著法，並保證合法（坑之三）。"""
        mapping = legal_indices(self.board)
        candidates = {i: c for i, c in root.children.items() if i in mapping}
        if not candidates:
            return self._choose_move()
        best_index = max(candidates, key=lambda i: candidates[i].visit_count)
        move = mapping[best_index]
        return move if move in self.board.legal_moves else self._choose_move()

    def _print_info(
        self,
        move: chess.Move,
        value: float,
        nodes: int,
        pv: list[chess.Move],
        elapsed_ms: int,
    ) -> None:
        """輸出 `info` 行（規格 §2.4）。

        **視角**：`score cp` 是**當前走棋方**視角，跟 value head 的輸出一致，
        所以這裡不用轉換（PGN 的 [%eval] 才需要轉成白方視角）。
        """
        # 走完這步是不是直接將死對方？是的話用 score mate 而不是 score cp
        score = f"cp {value_to_cp(value)}"
        if move != chess.Move.null():
            self.board.push(move)
            if self.board.is_checkmate():
                score = "mate 1"     # 正數 = 自己在 N 步內將死對方
            self.board.pop()

        nodes = max(nodes, 1)
        nps = int(nodes * 1000 / max(elapsed_ms, 1))
        # MCTS 的 depth 用 pv 長度；greedy + 送子檢查等於 depth 1、seldepth 2
        depth = max(len(pv), 1) if isinstance(self.searcher, MCTSSearcher) else 1
        seldepth = max(depth, 2)
        pv_text = " ".join(m.uci() for m in pv)
        print(
            f"info depth {depth} seldepth {seldepth} score {score} nodes {nodes} "
            f"nps {nps} time {elapsed_ms} pv {pv_text}"
        )

    # --- 主迴圈 -------------------------------------------------------------

    def run(self) -> None:
        """讀 stdin 跑協定迴圈，直到收到 quit 或 EOF。"""
        # 坑之一：Python 的 stdout 接到 pipe 時是「塊緩衝」，print 出去的字可能
        # 卡在緩衝區沒送出，GUI 等到超時就判引擎當機。一定要開 line buffering。
        # （UTF-8 已由 src/__init__.py 統一處理，這裡只補行緩衝。）
        sys.stdout.reconfigure(line_buffering=True)

        while True:
            try:
                # lstrip 掉 BOM：從 PowerShell 用管線餵指令時，第一行開頭會多一個
                # BOM，害 "uci" 比對不到。真正的 GUI 不會這樣，但擋掉很便宜。
                line = input().lstrip("﻿").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                continue

            parts = line.split()
            command = parts[0]

            try:
                if command == "uci":
                    self.handle_uci()
                elif command == "isready":
                    # 任何時候都可能收到，一律立刻回，不要做任何耗時的事
                    print("readyok", flush=True)
                elif command == "ucinewgame":
                    self.board = chess.Board()
                elif command == "setoption":
                    self.handle_setoption(parts)
                elif command == "position":
                    self.handle_position(parts)
                elif command == "go":
                    self.handle_go(parts)
                elif command == "stop":
                    # Phase 1 的搜尋是同步且瞬間完成的，收到 stop 時 bestmove
                    # 早就送出去了，所以這裡沒事可做。
                    # Phase 2 接上 MCTS 之後要改成獨立執行緒搜尋，stop 才有意義。
                    pass
                elif command == "quit":
                    return
                else:
                    # UCI 規定不認識的指令要直接忽略，不能報錯
                    log_stderr(f"[uci] 忽略不認識的指令：{line}")
            except Exception as exc:  # noqa: BLE001
                # 最外層的保險：任何 handler 出事都不能讓引擎整個死掉
                log_stderr(f"[uci] 處理「{line}」時發生例外：{type(exc).__name__}: {exc}")


def play_uci(cfg: Config, checkpoint: str, device: torch.device) -> None:
    """啟動 UCI 模式（可掛進 Cute Chess、Arena 等 GUI）。"""
    UciEngine(cfg, checkpoint, device).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="跟訓練好的 AI 下棋",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--mode", type=str, default="cli", choices=["cli", "uci"], help="對弈模式"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="模型 checkpoint 路徑"
    )
    parser.add_argument(
        "--black", action="store_true", help="人類執黑（預設執白，先走）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="0 = 每次都走最好的一步；> 0 = 依機率抽樣（棋風比較多變）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, preset=args.preset)
    if args.device:
        cfg.device = args.device

    if args.mode == "uci":
        # UCI 模式的模型載入在 UciEngine 裡面做，因為 setoption name Checkpoint
        # 可以中途換模型。這裡不能印任何東西到 stdout（坑之二）。
        play_uci(cfg, args.checkpoint, resolve_device(cfg.device))
        return

    searcher, model = load_searcher(args.checkpoint, cfg, temperature=args.temperature)
    print(f"已載入模型：{args.checkpoint}（{model.count_parameters():,} 參數）")
    play_cli(
        searcher,
        cfg,
        human_is_white=not args.black,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
