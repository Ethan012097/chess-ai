"""model.py 的測試。全部在 CPU 上跑，確保程式碼沒有寫死 CUDA。

最有價值的一條是 `test_overfit_small_batch`：對 8 筆資料訓練 200 步，loss 應該降到
0.1 以下。如果這條過不了，代表模型 / 損失函數 / 反向傳播接錯了，
訓練再久也不會有結果。
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from src.config import load_config
from src.dataset import ChessPositionDataset, build_dataloader
from src.encoding import (
    BOARD_SIZE,
    NUM_INPUT_PLANES,
    NUM_MOVES,
    encode_board,
    encode_board_compact,
)
from src.model import ChessNet, ResidualBlock, resolve_device
from src.preprocess import POSITION_DTYPE

DEVICE = torch.device("cpu")   # 所有測試都在 CPU 上跑
BATCH = 4


@pytest.fixture(scope="module")
def small_model() -> ChessNet:
    """小模型，測形狀用（跑得快）。"""
    return ChessNet(channels=16, blocks=2)


# --- 形狀與值域 -------------------------------------------------------------


def test_forward_shapes(small_model: ChessNet) -> None:
    """(4, 18, 8, 8) → (4, 4672) 與 (4, 1)。"""
    x = torch.randn(BATCH, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    policy_logits, value = small_model(x)
    assert policy_logits.shape == (BATCH, NUM_MOVES)
    assert value.shape == (BATCH, 1)


def test_value_output_range(small_model: ChessNet) -> None:
    """value 經過 tanh，必須落在 [-1, 1]。"""
    x = torch.randn(BATCH * 8, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE) * 10
    _, value = small_model(x)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_policy_logits_are_raw(small_model: ChessNet) -> None:
    """policy head 輸出 raw logits：不該已經是機率（總和不會是 1）。"""
    x = torch.randn(BATCH, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    policy_logits, _ = small_model(x)
    sums = policy_logits.exp().sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    assert torch.any(policy_logits < 0), "raw logits 應該有負值"


def test_residual_block_preserves_shape() -> None:
    """殘差塊輸入輸出同形狀。"""
    block = ResidualBlock(channels=16)
    x = torch.randn(2, 16, BOARD_SIZE, BOARD_SIZE)
    assert block(x).shape == x.shape


# --- 參數量 -----------------------------------------------------------------


# CLAUDE.md §1 的表格宣稱 small/base/large 分別約 3.5M / 7M / 22M 參數，但那個
# 數字跟同一份文件 §6 定義的架構對不起來：照 §6 的結構，C=128、N=10 算出來是
# 3.19M，每個 preset 都差了約 2.2 倍（要達到宣稱值，base 需要 23 個 block 而非 10）。
# 這裡以「§6 的架構 + §1 的 channels/blocks」為準（這兩者互相吻合，且 blocks 數
# 與 VRAM 建議是同一列資料），把測試範圍設成實際值附近，這樣才擋得住不小心改壞
# 架構的情況。若要改成宣稱的參數量，把 config.yaml 的 blocks 調成 20/23/33 即可。
EXPECTED_PARAM_COUNTS: dict[str, int] = {
    "small": 1_541_434,   # C=96,  N=8
    "base": 3_192_026,    # C=128, N=10
    "large": 9_591_322,   # C=192, N=14
}
PARAM_COUNT_TOLERANCE = 0.02   # 允許 2% 誤差，避免小改動就紅燈


def test_base_preset_parameter_count() -> None:
    """base preset（C=128, N=10）的參數量要符合 §6 架構算出來的值。"""
    cfg = load_config(preset="base")
    assert cfg.model.channels == 128 and cfg.model.blocks == 10
    n = ChessNet.from_config(cfg).count_parameters()
    expected = EXPECTED_PARAM_COUNTS["base"]
    assert abs(n - expected) / expected < PARAM_COUNT_TOLERANCE, (
        f"base preset 參數量 {n:,}，預期約 {expected:,}。架構被改動了嗎？"
    )


def test_all_presets_parameter_counts() -> None:
    """三個 preset 的參數量都要對，且 small < base < large。"""
    counts = {}
    for name in ("small", "base", "large"):
        cfg = load_config(preset=name)
        counts[name] = ChessNet.from_config(cfg).count_parameters()
        expected = EXPECTED_PARAM_COUNTS[name]
        assert abs(counts[name] - expected) / expected < PARAM_COUNT_TOLERANCE, (
            f"{name} preset 參數量 {counts[name]:,}，預期約 {expected:,}"
        )
    assert counts["small"] < counts["base"] < counts["large"]


def test_preset_overrides_batch_size() -> None:
    """preset 要能覆寫 batch_size。"""
    assert load_config(preset="small").train.batch_size == 512
    assert load_config(preset="base").train.batch_size == 1024
    assert load_config(preset="large").train.batch_size == 1536


# --- 過擬合測試：最重要的一條 ------------------------------------------------


def test_overfit_small_batch() -> None:
    """對 8 筆資料訓練 200 步，loss 應降到 0.1 以下。

    這條測試同時驗證了：模型接得起來、梯度流得動、policy 與 value 兩個頭都有在學。
    """
    torch.manual_seed(0)
    model = ChessNet(channels=32, blocks=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x = torch.randn(8, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    policy_target = torch.randint(0, NUM_MOVES, (8,), device=DEVICE)
    value_target = torch.tensor(
        [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0], device=DEVICE
    )

    model.train()
    loss = torch.tensor(float("inf"))
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        policy_logits, value = model(x)
        policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_target)
        value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), value_target)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"200 步後 loss 還有 {loss.item():.4f}，模型沒學起來"


def test_gradients_reach_both_heads() -> None:
    """policy head 與 value head 都要拿得到梯度（有一邊斷掉會很難發現）。"""
    model = ChessNet(channels=16, blocks=1)
    x = torch.randn(2, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    policy_logits, value = model(x)
    (policy_logits.sum() + value.sum()).backward()

    assert model.policy_conv.weight.grad is not None
    assert model.policy_conv.weight.grad.abs().sum() > 0
    assert model.value_fc2.weight.grad is not None
    assert model.value_fc2.weight.grad.abs().sum() > 0
    assert model.stem_conv.weight.grad.abs().sum() > 0


# --- 裝置處理 ---------------------------------------------------------------


def test_model_runs_on_cpu(small_model: ChessNet) -> None:
    """不可以寫死 cuda：模型要能完全在 CPU 上跑。"""
    model = small_model.to("cpu")
    x = torch.randn(2, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    policy_logits, value = model(x)
    assert policy_logits.device.type == "cpu" and value.device.type == "cpu"


def test_resolve_device_auto_falls_back_to_cpu() -> None:
    """device=auto 在沒有 CUDA 的機器上要回傳 cpu，不能爆掉。"""
    device = resolve_device("auto")
    assert device.type in ("cuda", "cpu")
    assert resolve_device("cpu").type == "cpu"


# --- checkpoint -------------------------------------------------------------


def test_from_checkpoint_round_trip(tmp_path: Path) -> None:
    """存檔再讀回來，權重與輸出要完全一致。"""
    cfg = load_config(preset="small")
    model = ChessNet.from_config(cfg)
    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 3,
            "global_step": 1234,
            "config": cfg.to_dict(),
        },
        ckpt_path,
    )

    loaded, checkpoint = ChessNet.from_checkpoint(ckpt_path, device="cpu")
    assert checkpoint["epoch"] == 3 and checkpoint["global_step"] == 1234
    assert loaded.channels == cfg.model.channels and loaded.blocks == cfg.model.blocks

    x = torch.randn(2, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        p1, v1 = model(x)
        p2, v2 = loaded(x)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(v1, v2)


def test_from_checkpoint_missing_file_has_helpful_error(tmp_path: Path) -> None:
    """錯誤訊息要告訴使用者下一步該跑什麼指令。"""
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        ChessNet.from_checkpoint(tmp_path / "nope.pt")


# --- dataset ----------------------------------------------------------------


def _write_fake_dataset(path: Path, num: int = 32) -> None:
    """用真實盤面產生一個小的 .npy 給 dataset 測試用。"""
    rows = []
    board = chess.Board()
    for i in range(num):
        if board.is_game_over():
            board = chess.Board()
        move = list(board.legal_moves)[0]
        pieces, castling, ep, halfmove = encode_board_compact(board)
        rows.append((pieces, castling, ep, halfmove, i % NUM_MOVES, (i % 3) - 1))
        board.push(move)
    arr = np.array(rows, dtype=POSITION_DTYPE)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    # np.save 會加上 .npy，如果原本就有就不會重複加
    if not path.exists() and path.with_suffix(".npy").exists():
        path.with_suffix(".npy").rename(path)


def test_dataset_returns_correct_shapes(tmp_path: Path) -> None:
    """Dataset 回傳 (18,8,8) float32、int64 index、float32 value。"""
    path = tmp_path / "train.npy"
    _write_fake_dataset(path)
    dataset = ChessPositionDataset(path)
    assert len(dataset) == 32

    board, policy, value = dataset[0]
    assert board.shape == (NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert board.dtype == torch.float32
    assert policy.dtype == torch.int64 and policy.dim() == 0
    assert value.dtype == torch.float32 and float(value) in (-1.0, 0.0, 1.0)


def test_dataset_decoding_matches_encoding(tmp_path: Path) -> None:
    """Dataset 展開出來的張量，必須跟 encode_board 完全一致。

    這條測試把「訓練時餵進網路的東西」跟「編碼定義」綁死，是最容易出錯又最難
    debug 的接縫。
    """
    path = tmp_path / "train.npy"
    _write_fake_dataset(path)
    dataset = ChessPositionDataset(path)

    board = chess.Board()
    for i in range(8):
        tensor, _, _ = dataset[i]
        np.testing.assert_array_equal(tensor.numpy(), encode_board(board))
        board.push(list(board.legal_moves)[0])


def test_dataset_soft_targets(tmp_path: Path) -> None:
    """soft_targets=True（Phase 2 用）要回傳 4672 維機率向量。"""
    path = tmp_path / "train.npy"
    _write_fake_dataset(path)
    dataset = ChessPositionDataset(path, soft_targets=True)
    _, policy, _ = dataset[5]
    assert policy.shape == (NUM_MOVES,)
    assert policy.sum() == pytest.approx(1.0)


def test_dataset_missing_file_has_helpful_error(tmp_path: Path) -> None:
    """錯誤訊息要指出該跑哪一行指令。"""
    with pytest.raises(FileNotFoundError, match="src.preprocess"):
        ChessPositionDataset(tmp_path / "nope.npy")


def test_dataloader_batches(tmp_path: Path) -> None:
    """DataLoader 疊出來的 batch 形狀要對，且能直接餵給模型。"""
    path = tmp_path / "train.npy"
    _write_fake_dataset(path)
    cfg = load_config(preset="small")
    cfg.train.num_workers = 0        # 測試不開 worker，Windows 上比較快
    loader = build_dataloader(path, cfg, shuffle=False, batch_size=8)

    boards, policies, values = next(iter(loader))
    assert boards.shape == (8, NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert policies.shape == (8,)
    assert values.shape == (8,)

    model = ChessNet(channels=16, blocks=1)
    policy_logits, value = model(boards)
    assert policy_logits.shape == (8, NUM_MOVES)
    assert value.shape == (8, 1)
