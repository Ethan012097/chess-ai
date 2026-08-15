"""自我對弈的測試（規格 §7）。

三件事最值得測，因為錯了都不會報錯、只會讓訓練悄悄變差：

1. **稀疏 policy 的 round-trip** —— 前 32 名壓縮再展開，機率要對得回去
2. **soft target 的損失公式** —— 規格寫的是
   `-(target * log_softmax(logits)).sum(dim=1).mean()`，
   我們用 `F.cross_entropy`。這條測試證明兩者數值相同，不是「應該相同」
3. **value target 的視角** —— 白勝的棋局裡，白方走的盤面要是 +1、黑方走的要是 -1

全部在 CPU 上跑，用隨機權重的小模型。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import load_config
from src.encoding import NUM_MOVES
from src.model import ChessNet
from src.preprocess import (
    PROB_QUANT_SCALE,
    SELFPLAY_DTYPE,
    SPARSE_POLICY_K,
    VALUE_QUANT_SCALE,
    decode_sparse_policy,
    encode_sparse_policy,
)
from src.selfplay import _final_white_score, play_one_game, sample_from_buffer, save_shard
from src.search.mcts import MCTSSearcher
from src.train import compute_loss

DEVICE = torch.device("cpu")


# --- 儲存格式 ---------------------------------------------------------------


def test_selfplay_dtype_is_about_200_bytes() -> None:
    """規格 §7.2 說約 200 bytes/盤面。實際是 197（不對齊、packed）。"""
    assert SELFPLAY_DTYPE.itemsize == 197
    assert SPARSE_POLICY_K == 32


def test_sparse_round_trip_preserves_distribution() -> None:
    """壓縮再展開，機率分佈要幾乎不變（量化誤差 < 1e-4）。"""
    rng = np.random.default_rng(0)
    indices = rng.choice(NUM_MOVES, size=20, replace=False)
    raw = rng.random(20)
    probs = {int(i): float(p / raw.sum()) for i, p in zip(indices, raw)}

    top_indices, top_probs, truncated = encode_sparse_policy(probs)
    dense = decode_sparse_policy(top_indices, top_probs, NUM_MOVES)

    assert truncated == pytest.approx(0.0), "只有 20 個著法，不該有任何截斷"
    assert dense.sum() == pytest.approx(1.0, abs=1e-6)
    for index, prob in probs.items():
        assert dense[index] == pytest.approx(prob, abs=1e-4)


def test_sparse_keeps_the_largest_k() -> None:
    """超過 K 個時要留下訪問次數最多的那些，不是隨便挑。"""
    probs = {i: (i + 1) / 5050.0 for i in range(100)}   # index 越大機率越高
    top_indices, top_probs, truncated = encode_sparse_policy(probs)

    kept = set(top_indices[top_probs > 0].tolist())
    assert kept == set(range(100 - SPARSE_POLICY_K, 100))
    # 被丟掉的是最小的 68 個
    expected_truncated = sum((i + 1) / 5050.0 for i in range(100 - SPARSE_POLICY_K))
    assert truncated == pytest.approx(expected_truncated, abs=1e-6)


def test_sparse_reports_truncated_mass() -> None:
    """截斷量是調整 K 的依據（§7.2：平均超過 0.02 就要調高），不能算錯。"""
    probs = {i: 1.0 / 40 for i in range(40)}            # 40 個等機率
    _, _, truncated = encode_sparse_policy(probs)
    # 留 32 個、丟 8 個 → 丟掉 8/40 = 0.2
    assert truncated == pytest.approx(8 / 40, abs=1e-6)


def test_sparse_renormalises_after_truncation() -> None:
    """展開後要重新正規化，否則損失的尺度會隨盤面浮動。"""
    probs = {i: 1.0 / 40 for i in range(40)}
    top_indices, top_probs, _ = encode_sparse_policy(probs)
    dense = decode_sparse_policy(top_indices, top_probs, NUM_MOVES)
    assert dense.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.count_nonzero(dense) == SPARSE_POLICY_K


def test_sparse_handles_empty_and_single() -> None:
    """沒有著法、只有一個著法都不能炸。"""
    ti, tp, tr = encode_sparse_policy({})
    assert tr == 0.0
    assert decode_sparse_policy(ti, tp, NUM_MOVES).sum() == 0.0

    ti, tp, tr = encode_sparse_policy({7: 1.0})
    dense = decode_sparse_policy(ti, tp, NUM_MOVES)
    assert dense[7] == pytest.approx(1.0)
    assert tp[0] == PROB_QUANT_SCALE


# --- 損失函數（規格 §7.2 的公式）-------------------------------------------


def test_soft_target_loss_matches_spec_formula() -> None:
    """`F.cross_entropy` 吃機率向量時，等於規格寫的那條公式。

    規格 §7.2：`-(target * log_softmax(logits)).sum(dim=1).mean()`
    這條測試的意義是：我們沒有「另外實作一個 soft target 分支」，
    而是證明既有的那一行本來就是規格要的東西。
    """
    torch.manual_seed(0)
    logits = torch.randn(8, NUM_MOVES)
    target = torch.rand(8, NUM_MOVES)
    target /= target.sum(dim=1, keepdim=True)

    spec = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    ours = F.cross_entropy(logits, target)
    assert ours.item() == pytest.approx(spec.item(), rel=1e-6)


def test_compute_loss_accepts_both_target_types() -> None:
    """同一個 compute_loss 要同時吃 index 與機率向量（兩種模式共用訓練迴圈）。"""
    torch.manual_seed(0)
    logits = torch.randn(4, NUM_MOVES)
    value = torch.randn(4, 1).tanh()
    value_target = torch.tensor([1.0, -1.0, 0.0, 1.0])

    hard = torch.tensor([3, 100, 4671, 0])
    soft = F.one_hot(hard, NUM_MOVES).float()

    loss_hard, p_hard, _ = compute_loss(logits, value, hard, value_target, 1.0)
    loss_soft, p_soft, _ = compute_loss(logits, value, soft, value_target, 1.0)

    # one-hot 的機率向量跟整數 index 是同一件事，損失必須相同
    assert p_hard.item() == pytest.approx(p_soft.item(), rel=1e-5)
    assert loss_hard.item() == pytest.approx(loss_soft.item(), rel=1e-5)


# --- 對局產生 ---------------------------------------------------------------


@pytest.fixture(scope="module")
def searcher() -> MCTSSearcher:
    torch.manual_seed(0)
    cfg = load_config(preset="small")
    model = ChessNet(channels=16, blocks=2)
    return MCTSSearcher(
        model, DEVICE, cfg, simulations=8, add_noise=True, batch_size=4
    )


def test_play_one_game_produces_valid_rows(searcher: MCTSSearcher) -> None:
    """一盤棋要產出可以直接寫進 SELFPLAY_DTYPE 的資料。"""
    rng = np.random.default_rng(0)
    game = play_one_game(searcher, temperature_moves=4, resign_threshold=-0.9,
                         allow_resign=True, rng=rng)

    assert game.plies > 0
    assert len(game.rows) == game.plies or game.resigned
    arr = np.array(game.rows, dtype=SELFPLAY_DTYPE)   # 塞不進去就會在這裡爆
    assert arr.dtype == SELFPLAY_DTYPE

    # 每個盤面至少要有一個著法的機率
    assert (arr["top_probs"].sum(axis=1) > 0).all()
    # value target 在 -1 ~ +1 的量化範圍內
    assert np.abs(arr["value_target"]).max() <= VALUE_QUANT_SCALE


def test_value_target_alternates_within_a_game(searcher: MCTSSearcher) -> None:
    """**視角測試**：同一盤棋裡相鄰兩步的 value target 必須互為相反數。

    白方走的盤面填白方的分數、黑方走的填黑方的分數。搞反的話網路會學到
    「讓對手變好是好事」，而且程式完全不會報錯。
    """
    rng = np.random.default_rng(1)
    for _ in range(5):
        game = play_one_game(searcher, 4, -0.9, True, rng)
        targets = [row[6] for row in game.rows]
        if game.result == 0:
            assert all(t == 0 for t in targets), "和局的 value target 應該全是 0"
        else:
            for i in range(len(targets) - 1):
                assert targets[i] == -targets[i + 1], "相鄰兩步的 target 沒有變號"
            return       # 找到一盤有勝負的就夠了


def test_final_white_score_from_resignation() -> None:
    """認輸的一方算輸。"""
    import chess

    board = chess.Board()
    assert _final_white_score(board, chess.WHITE) == -1
    assert _final_white_score(board, chess.BLACK) == 1
    # 沒結束又沒人認輸（撞到步數上限）→ 和局
    assert _final_white_score(board, None) == 0


def test_checkmate_scores_correctly() -> None:
    """自然結束時用 board.result()。"""
    import chess

    assert _final_white_score(chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"), None) == 1
    assert _final_white_score(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), None) == 0


# --- replay buffer ----------------------------------------------------------


def test_buffer_sampling_prefers_recent(tmp_path) -> None:
    """抽樣要偏向較新的 iteration（§7.4）。

    做法：舊檔全填 value=-10000、新檔全填 +10000，抽 2000 筆看平均。
    偏向新資料的話平均會明顯大於 0。
    """
    def make(path, value, n=500):
        rows = []
        for _ in range(n):
            rows.append(
                (np.zeros(64, np.int8), 0, -1, 0,
                 np.zeros(SPARSE_POLICY_K, np.uint16), np.zeros(SPARSE_POLICY_K, np.uint16), value)
            )
        save_shard(rows, path)

    make(tmp_path / "iter_0001.npy", -VALUE_QUANT_SCALE)
    make(tmp_path / "iter_0010.npy", VALUE_QUANT_SCALE)

    batch = sample_from_buffer(tmp_path, 2000, np.random.default_rng(0))
    assert len(batch) == 2000
    assert batch["value_target"].mean() > 0, "抽樣沒有偏向較新的 iteration"


def test_buffer_error_message_points_to_next_command(tmp_path) -> None:
    """buffer 空的時候要直接講下一步該跑什麼指令。"""
    with pytest.raises(FileNotFoundError, match="src.selfplay"):
        sample_from_buffer(tmp_path, 10, np.random.default_rng(0))


def test_prune_keeps_only_recent(tmp_path) -> None:
    """滑動視窗只留最近 N 代。"""
    from src.selfplay import buffer_files, prune_buffer

    for i in range(1, 26):
        save_shard(
            [(np.zeros(64, np.int8), 0, -1, 0,
              np.zeros(SPARSE_POLICY_K, np.uint16), np.zeros(SPARSE_POLICY_K, np.uint16), 0)],
            tmp_path / f"iter_{i:04d}.npy",
        )
    removed = prune_buffer(tmp_path, keep=20)
    assert len(removed) == 5
    remaining = [p.name for p in buffer_files(tmp_path)]
    assert remaining[0] == "iter_0006.npy"
    assert remaining[-1] == "iter_0025.npy"


# --- Dataset 讀自我對弈資料 -------------------------------------------------


def test_dataset_reads_selfplay_format(tmp_path) -> None:
    """`ChessPositionDataset` 要能直接吃自我對弈的檔案（train.py 一行都不用改）。"""
    from src.dataset import ChessPositionDataset

    probs = {10: 0.5, 20: 0.3, 30: 0.2}
    ti, tp, _ = encode_sparse_policy(probs)
    pieces = np.zeros(64, np.int8)
    pieces[0] = 4          # 己方的車
    save_shard([(pieces, 0b0011, -1, 5, ti, tp, 7500)], tmp_path / "iter_0001.npy")

    ds = ChessPositionDataset(tmp_path / "iter_0001.npy", soft_targets=True)
    board, policy, value = ds[0]

    assert board.shape == (18, 8, 8)
    assert policy.shape == (NUM_MOVES,)
    assert float(policy.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(policy[10]) == pytest.approx(0.5, abs=1e-3)
    assert float(value) == pytest.approx(0.75, abs=1e-4)


def test_dataset_rejects_selfplay_without_soft_targets(tmp_path) -> None:
    """自我對弈資料配 soft_targets=false 是設定錯誤，要當場講清楚。"""
    from src.dataset import ChessPositionDataset

    save_shard(
        [(np.zeros(64, np.int8), 0, -1, 0,
          np.zeros(SPARSE_POLICY_K, np.uint16), np.zeros(SPARSE_POLICY_K, np.uint16), 0)],
        tmp_path / "iter_0001.npy",
    )
    with pytest.raises(ValueError, match="soft_targets"):
        ChessPositionDataset(tmp_path / "iter_0001.npy", soft_targets=False)


def test_candidate_checkpoint_records_matching_architecture(tmp_path) -> None:
    """存候選模型時，記的架構必須跟權重一致。

    踩過的坑：直接寫 `cfg.to_dict()`，但 cfg 來自 config.yaml 的預設 preset
    （base, C128），而權重是從 best.pt 繼承的 small(C96)。
    下一代載入時會照 C128 建模型再去載 C96 的權重 → size mismatch。
    """
    import torch
    from src.model import ChessNet

    model = ChessNet(channels=16, blocks=2)
    ckpt_config = {"model": {"channels": 16, "blocks": 2,
                             "value_head_channels": 8, "value_hidden": 256}}
    path = tmp_path / "candidate.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "config": ckpt_config, "epoch": 1}, path)

    # 存什麼就要能載回什麼，不能靠全域設定去猜
    loaded, ck = ChessNet.from_checkpoint(path, device="cpu")
    assert loaded.count_parameters() == model.count_parameters()
    assert ck["config"]["model"]["channels"] == 16
