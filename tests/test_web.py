"""網頁棋盤後端的測試（規格 §4）。

最重要的是**視角測試**：評估條畫反了程式不會報錯，網頁看起來也很正常，
只是每一局都會顯示成優勢在錯的一方。這種 bug 只能靠測試抓。

技巧：不需要訓練好的模型也能測視角。因為盤面編碼是 canonical 的
（永遠翻轉成「輪到走的一方在下方」），所以「白方走的盤面」與「它的鏡射」
餵進網路會得到**完全相同**的 value。於是：

    value（走棋方視角）  應該相同
    value_white（白方視角）應該正負相反

只要後端有一個地方少取或多取一次負號，這條就會爆。全部用隨機權重的小模型在
CPU 上跑，跟棋力無關。
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch
from fastapi.testclient import TestClient

from src.config import load_config
from src.model import ChessNet
from src.search.greedy import GreedySearcher
from src.web import server

DEVICE = torch.device("cpu")

# 白方多一個后，明顯優勢
WHITE_WINNING_FEN = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
# 黑方已被將死（白方 Ra8#）
CHECKMATED_FEN = "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


@pytest.fixture(scope="module", autouse=True)
def engine_state() -> server.EngineState:
    """塞一個隨機權重的小模型進去，測試不需要真的有 models/best.pt。"""
    torch.manual_seed(0)
    cfg = load_config(preset="small")
    model = ChessNet(channels=16, blocks=2)
    state = server.EngineState(
        cfg=cfg,
        device=DEVICE,
        model=model,
        checkpoint_path=Path("models/test-fake.pt"),
        epoch=3,
        use_mcts=False,
        simulations=16,
        searcher=GreedySearcher(model, DEVICE, cfg),
    )
    server.install_state(state)
    yield state
    # 測試之間不要互相污染
    server.install_state(None)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server.app)


# --- 視角（最重要的一組）----------------------------------------------------


def test_value_white_flips_on_mirrored_position(client: TestClient) -> None:
    """同一個盤面鏡射之後，走棋方視角的 value 不變、白方視角的 value 要變號。

    這是評估條畫反與否的決定性測試。
    """
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mirrored = board.mirror()

    a = client.post("/api/analyse", json={"fen": board.fen()}).json()
    b = client.post("/api/analyse", json={"fen": mirrored.fen()}).json()

    # canonical 編碼 → 網路看到的是同一件事
    assert a["value"] == pytest.approx(b["value"], abs=1e-4)
    # 但白方視角必須相反
    assert a["value_white"] == pytest.approx(-b["value_white"], abs=1e-4)
    assert a["cp"] == -b["cp"]


def test_value_white_equals_value_when_white_to_move(client: TestClient) -> None:
    """白方走棋時，兩個視角本來就是同一個數字。"""
    res = client.post("/api/analyse", json={"fen": chess.STARTING_FEN}).json()
    assert res["value"] == pytest.approx(res["value_white"])


def test_value_white_is_negated_when_black_to_move(client: TestClient) -> None:
    """黑方走棋時，白方視角要取負號。"""
    board = chess.Board()
    board.push_san("e4")
    res = client.post("/api/analyse", json={"fen": board.fen()}).json()
    assert res["value_white"] == pytest.approx(-res["value"])


def test_checkmate_value_is_loss_for_side_to_move(client: TestClient) -> None:
    """被將死的一方，走棋方視角是 -1；這裡輪到黑方，所以白方視角是 +1。"""
    res = client.post("/api/analyse", json={"fen": CHECKMATED_FEN}).json()
    assert res["is_game_over"] is True
    assert res["result"] == "1-0"
    assert res["result_reason"] == "將死"
    assert res["value"] == pytest.approx(-1.0)
    assert res["value_white"] == pytest.approx(1.0)
    assert res["moves"] == []
    assert res["best"] is None


# --- /api/analyse -----------------------------------------------------------


def test_analyse_returns_spec_fields(client: TestClient) -> None:
    """回應要有規格 §4.2 列的全部欄位。"""
    res = client.post("/api/analyse", json={"fen": chess.STARTING_FEN, "top_k": 5}).json()
    for key in ("value", "value_white", "cp", "moves", "is_game_over", "result", "elapsed_ms"):
        assert key in res, f"回應少了 {key}"
    assert len(res["moves"]) == 5
    for m in res["moves"]:
        assert set(m) == {"uci", "san", "prob", "visits"}


def test_analyse_moves_are_legal_and_sorted(client: TestClient) -> None:
    """候選著法必須合法，而且依機率由高到低排序（前端畫箭頭時直接吃順序）。"""
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    res = client.post("/api/analyse", json={"fen": board.fen(), "top_k": 8}).json()

    probs = [m["prob"] for m in res["moves"]]
    assert probs == sorted(probs, reverse=True)
    for m in res["moves"]:
        assert chess.Move.from_uci(m["uci"]) in board.legal_moves
        assert board.san(chess.Move.from_uci(m["uci"])) == m["san"]
        assert 0.0 <= m["prob"] <= 1.0


def test_analyse_top_k_is_respected(client: TestClient) -> None:
    """top_k 決定回傳幾個候選。"""
    for k in (1, 3, 8):
        res = client.post("/api/analyse", json={"fen": chess.STARTING_FEN, "top_k": k}).json()
        assert len(res["moves"]) == k


def test_analyse_rejects_out_of_range_top_k(client: TestClient) -> None:
    """前端滑桿是 1–8，超出範圍要被擋掉而不是默默截斷。"""
    assert client.post("/api/analyse", json={"fen": chess.STARTING_FEN, "top_k": 9}).status_code == 422
    assert client.post("/api/analyse", json={"fen": chess.STARTING_FEN, "top_k": 0}).status_code == 422


def test_analyse_rejects_bad_fen(client: TestClient) -> None:
    """FEN 壞掉要回 400 並說明正確格式，不是 500。"""
    res = client.post("/api/analyse", json={"fen": "這不是 FEN"})
    assert res.status_code == 400
    assert "FEN" in res.json()["detail"]


def test_analyse_best_move_is_legal(client: TestClient) -> None:
    """`best` 是 AI 真的會走的那一步（含將死檢查與送子檢查），必須合法。"""
    board = chess.Board(WHITE_WINNING_FEN)
    res = client.post("/api/analyse", json={"fen": board.fen()}).json()
    assert chess.Move.from_uci(res["best"]["uci"]) in board.legal_moves
    assert res["source"] == "policy"


# --- /api/move --------------------------------------------------------------


def test_move_applies_legal_move(client: TestClient) -> None:
    res = client.post("/api/move", json={"fen": chess.STARTING_FEN, "uci": "e2e4"}).json()
    assert res["legal"] is True
    assert res["san"] == "e4"
    assert res["fen"].startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")
    assert res["is_game_over"] is False


def test_move_rejects_illegal_move_without_changing_fen(client: TestClient) -> None:
    """非法著法要回傳 legal:false 與**原本的** FEN，前端據此把棋子彈回去。"""
    res = client.post("/api/move", json={"fen": chess.STARTING_FEN, "uci": "e2e5"}).json()
    assert res["legal"] is False
    assert res["fen"] == chess.STARTING_FEN


def test_move_rejects_garbage_uci(client: TestClient) -> None:
    """看不懂的字串也要好好回答，不能讓伺服器 500。"""
    res = client.post("/api/move", json={"fen": chess.STARTING_FEN, "uci": "zz99"}).json()
    assert res["legal"] is False
    assert "看不懂" in res["reason"]


def test_move_reports_checkmate(client: TestClient) -> None:
    """走出將死時要回報結果，自動播放才知道要停。"""
    res = client.post(
        "/api/move", json={"fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "uci": "a1a8"}
    ).json()
    assert res["legal"] is True
    assert res["is_game_over"] is True
    assert res["result"] == "1-0"
    assert res["result_reason"] == "將死"


def test_move_handles_promotion(client: TestClient) -> None:
    """升變的 UCI 是 5 個字元（前端一律送 q）。"""
    res = client.post("/api/move", json={"fen": "8/P6k/8/8/8/8/8/6K1 w - - 0 1", "uci": "a7a8q"}).json()
    assert res["legal"] is True
    assert res["san"] == "a8=Q"      # a8 與黑王的 h7 不同線，所以沒有將軍記號
    assert res["fen"].startswith("Q7/7k/")


# --- /api/health ------------------------------------------------------------


def test_health_reports_checkpoint(client: TestClient) -> None:
    """要看得出載入的是哪個 checkpoint，方便比較不同版本。"""
    res = client.get("/api/health").json()
    assert res["model_path"].endswith("test-fake.pt")
    assert res["device"] == "cpu"
    assert res["params"] > 0
    assert res["epoch"] == 3
    assert res["searcher"] == "policy"


# --- /api/pgn ---------------------------------------------------------------


def test_pgn_export_has_eval_comments(client: TestClient) -> None:
    """匯出的 PGN 要帶 lichess 認得的 [%eval]，而且視角是白方。"""
    res = client.post(
        "/api/pgn",
        json={
            "moves": [
                {"uci": "e2e4", "value": 0.20, "policy_top": [["e2e4", 0.4]], "elapsed_ms": 30},
                # 黑方走棋，value 是黑方視角的 +0.10 → PGN 應寫成負值
                {"uci": "e7e5", "value": 0.10, "policy_top": [["e7e5", 0.35]], "elapsed_ms": 28},
            ],
            "headers": {"White": "Human", "Black": "MyNet"},
        },
    ).json()

    pgn = res["pgn"]
    assert '[White "Human"]' in pgn
    assert "[%eval " in pgn
    assert "1. e4" in pgn and "e5" in pgn
    # 黑方那一步的評分必須是負的（白方視角）
    evals = [line for line in pgn.split("[%eval ")[1:]]
    assert not evals[0].startswith("-"), "白方 +0.20 卻寫成負值，視角反了"
    assert evals[1].startswith("-"), "黑方視角 +0.10 應該寫成白方視角的負值"


def test_pgn_export_rejects_illegal_sequence(client: TestClient) -> None:
    """著法串不合法時要明講是第幾步，不要丟一個看不懂的例外。"""
    res = client.post("/api/pgn", json={"moves": [{"uci": "e2e4"}, {"uci": "e2e4"}]})
    assert res.status_code == 400
    assert "第 2 步" in res.json()["detail"]


# --- /api/config（切換搜尋方式）---------------------------------------------


def test_switching_to_mcts_changes_arrow_source(client: TestClient) -> None:
    """切成 MCTS 之後，箭頭資料源要從 policy 機率換成訪問次數比例（§4.4）。"""
    try:
        health = client.post("/api/config", json={"mcts": True, "simulations": 24}).json()
        assert health["searcher"] == "mcts"
        assert health["simulations"] == 24

        res = client.post("/api/analyse", json={"fen": chess.STARTING_FEN, "top_k": 5}).json()
        assert res["source"] == "mcts"
        assert all(m["visits"] is not None for m in res["moves"])
        assert sum(m["visits"] for m in res["moves"]) > 0
        # 訪問次數比例仍然是機率，總和不會超過 1
        assert sum(m["prob"] for m in res["moves"]) <= 1.0 + 1e-6
    finally:
        client.post("/api/config", json={"mcts": False})


def test_policy_mode_has_no_visits(client: TestClient) -> None:
    """policy 模式下 visits 必須是 null，前端才知道要用 prob 畫箭頭。"""
    res = client.post("/api/analyse", json={"fen": chess.STARTING_FEN}).json()
    assert res["source"] == "policy"
    assert all(m["visits"] is None for m in res["moves"])


# --- 靜態頁面 ---------------------------------------------------------------


def test_index_page_is_served(client: TestClient) -> None:
    """`GET /` 要回傳前端頁面本體（同源，不需要 CORS）。"""
    res = client.get("/")
    assert res.status_code == 200
    assert "chess-ai" in res.text
    assert "chessboard" in res.text


def test_index_html_has_no_build_step() -> None:
    """規格 §4.1：前端只有這一個檔，不能出現 npm / webpack 的痕跡。"""
    html = server.INDEX_HTML.read_text(encoding="utf-8")
    assert "import " not in html.split("<script>")[-1] or "importScripts" in html
    assert len(list(server.STATIC_DIR.iterdir())) == 1, "static/ 應該只有 index.html 一個檔"


# --- 小工具 -----------------------------------------------------------------


def test_value_payload_perspective() -> None:
    """`value_payload` 是全專案唯一做視角轉換的地方，單獨測一次。"""
    white = server.value_payload(0.5, chess.WHITE)
    black = server.value_payload(0.5, chess.BLACK)
    assert white["value_white"] == pytest.approx(0.5)
    assert black["value_white"] == pytest.approx(-0.5)
    assert white["cp"] == -black["cp"]
    assert white["cp"] > 0


def test_game_over_info_reasons() -> None:
    """各種結束方式都要給得出中文說明。"""
    assert server.game_over_info(chess.Board())[0] is False
    over, result, reason = server.game_over_info(chess.Board(CHECKMATED_FEN))
    assert (over, result, reason) == (True, "1-0", "將死")
    over, result, reason = server.game_over_info(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert (over, result) == (True, "1/2-1/2")
    assert "逼和" in reason
    over, result, reason = server.game_over_info(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    assert (over, result, reason) == (True, "1/2-1/2", "子力不足")
