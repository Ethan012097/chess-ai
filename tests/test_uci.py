"""UCI 協定測試：用 subprocess 啟動自己的引擎，不需要開 GUI 就能驗證。

這幾條測試守住的是「Cute Chess 載得起來嗎」。UCI 出問題時 GUI 通常只會說
「引擎沒回應」，不會告訴你原因，所以在這裡先抓出來便宜太多了。

**每個測試都設 timeout**，避免引擎 hang 住時把整個測試流程卡死。

測試一律用 `--device cpu` 與臨時建立的隨機權重 checkpoint：
不依賴訓練成果，剛 clone 下來就能跑。
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

import chess
import pytest
import torch

from src.config import PROJECT_ROOT, load_config
from src.model import ChessNet

# 啟動要 import torch 並載入模型，Windows 上這段可能要好幾秒，所以握手給寬一點。
# 規格說「5 秒內收到 uciok」指的是引擎就緒之後的反應速度，不是含 Python 啟動時間。
STARTUP_TIMEOUT = 90.0
# 引擎就緒之後，每個指令的反應都應該很快
COMMAND_TIMEOUT = 20.0
QUIT_TIMEOUT = 10.0

BACK_RANK_MATE_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# 黑方已被底線將死（Ra8#），無合法著法。
# 註：Rh1 那種寫法不會將死——h7 的兵擋住了直線，黑方還有 8 步可走。
CHECKMATED_FEN = "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"          # 黑方逼和，無合法著法


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """建立一個隨機權重的小 checkpoint 給測試用。

    用真的訓練成果會讓測試依賴前面的步驟，剛 clone 下來就跑不了。
    UCI 協定本身跟棋力無關，隨機權重完全夠測。
    """
    cfg = load_config(preset="small")
    model = ChessNet.from_config(cfg)
    path = tmp_path_factory.mktemp("uci") / "random.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "global_step": 0,
            "config": cfg.to_dict(),
        },
        path,
    )
    return path


class UciProcess:
    """包一層 subprocess，提供「讀一行、但最多等 N 秒」的能力。

    為什麼需要這個 class：`stdout.readline()` 會一直卡住，而 Windows 上不能對
    pipe 用 select()。所以開一條背景執行緒把每一行丟進 Queue，主執行緒就能用
    `Queue.get(timeout=...)` 做有時限的讀取。
    """

    def __init__(self, checkpoint: Path) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.play",
                "--mode",
                "uci",
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "small",
                "--device",
                "cpu",
            ],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        """背景執行緒：把 stdout 的每一行丟進 queue。"""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.strip())

    def send(self, command: str) -> None:
        """送一行指令給引擎。"""
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, prefix: str, timeout: float = COMMAND_TIMEOUT) -> str:
        """一直讀到某個開頭的行為止。

        Args:
            prefix: 要等的行開頭，例如 "uciok"、"bestmove"。
            timeout: 總共最多等幾秒。

        Returns:
            符合的那一行。

        Raises:
            AssertionError: 逾時還沒等到（附上已收到的內容與 stderr，方便除錯）。
        """
        collected: list[str] = []
        deadline = threading.Event()
        timer = threading.Timer(timeout, deadline.set)
        timer.start()
        try:
            while not deadline.is_set():
                try:
                    line = self.lines.get(timeout=0.1)
                except queue.Empty:
                    if self.process.poll() is not None:
                        break
                    continue
                collected.append(line)
                if line.startswith(prefix):
                    return line
        finally:
            timer.cancel()

        raise AssertionError(
            f"等不到開頭為 {prefix!r} 的回應（{timeout} 秒）。\n"
            f"已收到：{collected}\n"
            f"引泣是否還活著：{self.process.poll() is None}"
        )

    def handshake(self) -> None:
        """送 uci 並等 uciok，確認引擎已就緒。"""
        self.send("uci")
        self.read_until("uciok", timeout=STARTUP_TIMEOUT)

    def close(self) -> None:
        """送 quit 並確保行程結束。"""
        try:
            if self.process.poll() is None:
                self.send("quit")
                self.process.wait(timeout=QUIT_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            self.process.kill()
        finally:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


@pytest.fixture
def engine(checkpoint: Path):
    """啟動引擎並在測試結束後關掉。"""
    proc = UciProcess(checkpoint)
    try:
        yield proc
    finally:
        proc.close()


# --- 規格 §2.7 的五條 -------------------------------------------------------


def test_uci_handshake_returns_uciok(engine: UciProcess) -> None:
    """送 uci，要收到 id name / id author 與 uciok。"""
    engine.send("uci")
    collected: list[str] = []
    deadline = threading.Event()
    timer = threading.Timer(STARTUP_TIMEOUT, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            try:
                line = engine.lines.get(timeout=0.1)
            except queue.Empty:
                continue
            collected.append(line)
            if line.startswith("uciok"):
                break
    finally:
        timer.cancel()

    assert any(line.startswith("uciok") for line in collected), f"沒收到 uciok：{collected}"
    assert any(line.startswith("id name") for line in collected), "缺少 id name"
    assert any(line.startswith("id author") for line in collected), "缺少 id author"


def test_isready_returns_readyok(engine: UciProcess) -> None:
    """送 isready，要收到 readyok。"""
    engine.handshake()
    engine.send("isready")
    assert engine.read_until("readyok").startswith("readyok")


def test_go_returns_legal_move_from_startpos(engine: UciProcess) -> None:
    """position startpos + go movetime 100 → bestmove 必須是起始盤面的合法著法。"""
    engine.handshake()
    engine.send("position startpos")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove")

    uci = line.split()[1]
    move = chess.Move.from_uci(uci)
    assert move in chess.Board().legal_moves, f"{uci} 不是起始盤面的合法著法"


def test_go_from_fen_with_moves(engine: UciProcess) -> None:
    """position fen ... moves ... 之後的 bestmove 也要合法。"""
    engine.handshake()
    engine.send("position startpos moves e2e4 e7e5 g1f3")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove")

    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
    assert chess.Move.from_uci(line.split()[1]) in board.legal_moves


def test_checkmate_position_does_not_crash_or_hang(engine: UciProcess) -> None:
    """送一個已經將死的盤面，引擎不可以崩潰也不可以 hang。

    沒有合法著法時，UCI 的慣例是回 `bestmove 0000`。
    """
    engine.handshake()
    engine.send(f"position fen {CHECKMATED_FEN}")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove")
    assert line.strip() == "bestmove 0000", f"預期 bestmove 0000，收到 {line!r}"

    # 之後還要能正常服務下一個指令（確認沒有進入壞掉的狀態）
    engine.send("isready")
    assert engine.read_until("readyok").startswith("readyok")


def test_stalemate_position_does_not_crash(engine: UciProcess) -> None:
    """逼和盤面同樣不能崩潰。"""
    engine.handshake()
    engine.send(f"position fen {STALEMATE_FEN}")
    engine.send("go movetime 100")
    assert engine.read_until("bestmove").strip() == "bestmove 0000"


def test_claimable_draw_still_returns_a_move(engine: UciProcess) -> None:
    """**可宣告的和棋不能回 0000。**

    三次重複與五十步規則是「可以宣告」而不是「自動成立」，盤面上還有合法著法。
    回 0000 的話 Cute Chess 會判定 illegal move 直接判負 ——
    這個 bug 曾經讓一場 80 局的引擎對打有 28 局是這樣輸掉的。

    這裡用五十步規則（halfmove clock = 100），因為它只靠 FEN 就能重現。
    """
    engine.handshake()
    fen = "8/8/4k3/8/8/4K3/6R1/8 w - - 100 200"
    board = chess.Board(fen)
    assert board.can_claim_fifty_moves(), "測試前提：這個盤面可以宣告五十步和棋"
    assert board.is_game_over(claim_draw=True), "測試前提：claim_draw=True 會說已結束"
    assert any(board.legal_moves), "測試前提：但盤面上還有合法著法"

    engine.send(f"position fen {fen}")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove").strip()
    assert line != "bestmove 0000", "可宣告和棋的盤面回了 0000，對打時會被判負"
    assert chess.Move.from_uci(line.split()[1]) in board.legal_moves


def test_repetition_position_still_returns_a_move(engine: UciProcess) -> None:
    """三次重複盤面（用 position startpos moves 重現）也要照常走棋。"""
    engine.handshake()
    # 兩邊各把馬走出去再走回來兩次 → 起始盤面出現第三次
    moves = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"
    board = chess.Board()
    for uci in moves.split():
        board.push_uci(uci)
    assert board.can_claim_threefold_repetition(), "測試前提：可以宣告三次重複"

    engine.send(f"position startpos moves {moves}")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove").strip()
    assert line != "bestmove 0000", "三次重複盤面回了 0000，對打時會被判負"
    assert chess.Move.from_uci(line.split()[1]) in board.legal_moves


def test_quit_terminates_process(checkpoint: Path) -> None:
    """送 quit，行程要在時限內自己結束。"""
    proc = UciProcess(checkpoint)
    try:
        proc.handshake()
        proc.send("quit")
        returncode = proc.process.wait(timeout=QUIT_TIMEOUT)
        assert returncode == 0, f"quit 之後回傳碼是 {returncode}"
    finally:
        proc.close()


# --- 額外的健壯性測試 -------------------------------------------------------


def test_finds_mate_in_one(engine: UciProcess) -> None:
    """底線將殺：即使是隨機權重，一步將死檢查也該找到 a1a8。

    這條同時驗證了 info 行會輸出 `score mate 1`。
    """
    engine.handshake()
    engine.send(f"position fen {BACK_RANK_MATE_FEN}")
    engine.send("go movetime 100")

    info_lines: list[str] = []
    while True:
        line = engine.lines.get(timeout=COMMAND_TIMEOUT)
        if line.startswith("bestmove"):
            assert line.split()[1] == "a1a8", f"沒找到一步將死，走了 {line}"
            break
        if line.startswith("info"):
            info_lines.append(line)

    assert any("mate 1" in line for line in info_lines), (
        f"info 行應該回報 score mate 1，實際：{info_lines}"
    )


def test_setoption_does_not_break_engine(engine: UciProcess) -> None:
    """setoption 之後引擎要照常運作（包含還沒實作的 MCTS 選項）。"""
    engine.handshake()
    for command in (
        "setoption name Temperature value 50",
        "setoption name TopK value 3",
        "setoption name MCTS value true",
        "setoption name Simulations value 1600",
        "setoption name NoSuchOption value 123",
    ):
        engine.send(command)

    engine.send("isready")
    assert engine.read_until("readyok").startswith("readyok")

    engine.send("position startpos")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove")
    assert chess.Move.from_uci(line.split()[1]) in chess.Board().legal_moves


def test_unknown_command_is_ignored(engine: UciProcess) -> None:
    """UCI 規定不認識的指令要直接忽略，不能報錯或結束。"""
    engine.handshake()
    engine.send("this is not a real uci command")
    engine.send("isready")
    assert engine.read_until("readyok").startswith("readyok")


def test_ucinewgame_resets_board(engine: UciProcess) -> None:
    """ucinewgame 之後盤面要回到起始狀態。"""
    engine.handshake()
    engine.send("position startpos moves e2e4 e7e5 g1f3 b8c6")
    engine.send("ucinewgame")
    engine.send("go movetime 100")
    line = engine.read_until("bestmove")
    assert chess.Move.from_uci(line.split()[1]) in chess.Board().legal_moves


def test_stdout_contains_only_protocol_lines(engine: UciProcess) -> None:
    """stdout 只能有 UCI 訊息（坑之二）。

    任何 tqdm 進度條、PyTorch 警告、除錯輸出跑到 stdout 都會讓 GUI 解析失敗。
    """
    engine.handshake()
    engine.send("position startpos")
    engine.send("go movetime 100")
    engine.read_until("bestmove")

    allowed = ("id ", "option ", "uciok", "readyok", "info ", "bestmove ")
    seen: list[str] = []
    while not engine.lines.empty():
        seen.append(engine.lines.get_nowait())
    for line in seen:
        assert line.startswith(allowed), f"stdout 出現非協定內容：{line!r}"
