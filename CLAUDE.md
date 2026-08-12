# CLAUDE.md — 西洋棋 AI 專案（Supervised Learning，GPU）

這份文件是本專案的**唯一規格書**。請依照這裡的定義產生完整、可直接執行的程式碼。
若本文件與你的既有習慣衝突，以本文件為準；若有本文件未定義的細節，選擇最簡單、最容易除錯的做法，並在程式碼註解中說明你的選擇。

---

## 0. 專案目標與兩階段規劃

**Phase 1（本次要完成，全部實作）**
用人類高分棋局做監督式學習，訓練一個「雙頭神經網路」：

- **Policy head**：給定盤面，預測人類會走哪一步
- **Value head**：給定盤面，預測這盤棋最後誰會贏（-1 ~ +1）

然後用這個網路直接下棋（policy 取最高分的合法著法），目標是能穩定打贏隨機走法、並與 Stockfish Skill Level 0~3 有得打。

**Phase 2（本次只留介面與 TODO，不實作）**
把同一個網路接上 MCTS + 自我對弈，變成 AlphaZero 簡化版。

**因此 Phase 1 的所有設計決定都必須是 Phase 2 相容的**，具體來說：

- 盤面編碼採用 **canonical orientation**（永遠翻轉成「輪到走的一方在下方」）
- 著法編碼採用 **AlphaZero 的 4672 維（73 planes × 64 squares）**
- 網路輸出 **policy logits + value**，Phase 2 直接沿用，不需重寫
- 訓練迴圈吃的是 `(encoded_board, policy_target, value_target)`，Phase 2 只是把 `policy_target` 從 one-hot 換成 MCTS 訪問次數分佈

---

## 1. 硬體與環境（GPU 訓練）

環境：

- Windows 11 + PowerShell
- Python 3.12 虛擬環境
- 專案根目錄：`C:\code\chess-ai`
- PyTorch 安裝指令（CUDA 版，寫進 README）：
  `pip install torch --index-url https://download.pytorch.org/whl/cu126`
- README 第一步要有這行自檢：
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  若 `cuda.is_available()` 是 False，要在 README 說明多半是裝到 CPU 版 wheel，需先 `pip uninstall torch` 再重裝。

### 裝置處理

- `config.yaml` 有 `device: auto`，程式碼 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- 模型、資料都 `.to(device)`；**不要在任何地方寫死 `cuda`**，測試要能在 CPU 上跑
- 啟用混合精度：`torch.amp.autocast("cuda", dtype=torch.bfloat16)` + `GradScaler`（bf16 不需要 scaler，但若顯卡是 GTX 10/16 系列不支援 bf16，就改 fp16 + scaler；請在程式碼中自動偵測 `torch.cuda.is_bf16_supported()`）
- 開 `torch.backends.cudnn.benchmark = True`（輸入形狀固定，能加速）
- 資料前處理只做一次，存成緊湊的二進位檔，訓練時用 `np.memmap` 讀，不要每個 epoch 重新解析 PGN
- `DataLoader` 用 `num_workers=4`、`pin_memory=True`、`persistent_workers=True`。Windows 上 worker 會 re-import 主模組，所以 `train.py` 的進入點**必須**包在 `if __name__ == "__main__":` 裡
- GPU 訓練時瓶頸通常在資料供給端，`__getitem__` 的解碼要用純 NumPy 向量化，不要在裡面呼叫 `python-chess`

### 依 VRAM 選 preset（寫進 `config.yaml` 的 `presets:` 區塊，用 `--preset` 選）

| preset | 適用 VRAM | channels | blocks | batch_size | 參數量 |
|---|---|---|---|---|---|
| `small` | 4–6 GB | 96 | 8 | 512 | ~3.5 M |
| `base` | 8–12 GB（**預設**） | 128 | 10 | 1024 | ~7 M |
| `large` | 16 GB 以上 | 192 | 14 | 1536 | ~22 M |

若訓練中出現 `CUDA out of memory`，錯誤處理要直接建議「換小一號的 preset，或把 batch_size 減半」。

---

## 2. 目錄結構（請完全照這個建立）

```
chess-ai/
├── CLAUDE.md
├── README.md                  # 安裝、逐步執行指令、預期輸出
├── requirements.txt
├── config.yaml                # 所有超參數集中在這
├── data/
│   ├── raw/                   # 下載的 .pgn / .pgn.zst（不進 git）
│   └── processed/             # 前處理後的 .npy memmap（不進 git）
├── models/                    # checkpoint（不進 git）
├── logs/                      # 訓練曲線 csv
├── src/
│   ├── __init__.py
│   ├── config.py              # 讀 config.yaml → dataclass
│   ├── encoding.py            # 盤面編碼、著法編碼（Phase 1/2 共用）
│   ├── preprocess.py          # PGN → .npy shards
│   ├── dataset.py             # torch Dataset / DataLoader
│   ├── model.py               # 雙頭網路
│   ├── train.py               # 訓練迴圈
│   ├── evaluate.py            # 準確率 + 對局測試 + Elo 估計
│   ├── play.py                # CLI 對弈 / UCI 介面
│   └── search/
│       ├── __init__.py
│       ├── greedy.py          # Phase 1：policy + 合法著法 mask
│       └── mcts.py            # Phase 2：只寫介面與 TODO
├── scripts/
│   └── download_data.py       # 下載 + 解壓資料
└── tests/
    ├── test_encoding.py
    ├── test_preprocess.py
    └── test_model.py
```

`.gitignore` 要排除 `data/`、`models/`、`logs/`、`.venv/`、`__pycache__/`。

---

## 3. 資料來源與下載

`scripts/download_data.py` 要支援下列來源，用 `--source` 參數選擇，預設 `elite`：

| source | URL | 說明 |
|---|---|---|
| `elite` | https://database.nikonoel.fr/ | **預設。**已篩選過 2000+ 分的 Lichess 棋局，單月約 100–400 MB，適合 CPU 訓練 |
| `lichess` | https://database.lichess.org/ | 完整月檔，`.pgn.zst`，單月可達 30 GB，需自行篩選。用 `zstandard` **串流解壓，不要整檔解到硬碟** |
| `ccrl` | https://computerchess.org.uk/4040/games.html | 引擎對局，PGN 內含每步評分，Phase 2 之後可拿來當 value 的額外標註 |
| `pgnmentor` | https://www.pgnmentor.com/files.html | 大師棋譜，檔案小、乾淨，適合快速煙霧測試 |

要求：

- 支援 `--month 2024-01` 之類的參數
- 下載要有進度條（`tqdm`）與續傳（檢查檔案已存在就跳過）
- `.zst` 一律用 `zstandard` 的 stream reader 邊解壓邊丟給解析器
- 另外提供 `--sample` 模式：只抓前 5000 局，讓整條 pipeline 可以在 5 分鐘內跑通一次

Stockfish（評估用）不要自動下載，在 README 指示使用者到 https://stockfishchess.org/download/ 抓 Windows 版，放到 `bin/stockfish.exe`，路徑寫在 `config.yaml`。

---

## 4. 盤面編碼（`src/encoding.py`）

這是整個專案最容易寫錯的地方，請寫得非常小心，並用 `tests/test_encoding.py` 驗證。

### 4.1 Canonical orientation

**永遠從「輪到走棋的一方」的視角編碼。** 如果輪到黑方，先把棋盤上下鏡射並交換雙方顏色（`python-chess` 的 `board.mirror()`），使得「自己」永遠是白方、永遠從第 0 排往上前進。

這樣網路只需要學會一種視角，資料效率加倍，而且 Phase 2 的 MCTS 也能直接用。

### 4.2 輸入張量：`(18, 8, 8)` float32

| planes | 內容 |
|---|---|
| 0–5 | 己方的 P, N, B, R, Q, K（各一層 0/1） |
| 6–11 | 對方的 P, N, B, R, Q, K |
| 12 | 己方王翼可易位（整層填 0 或 1） |
| 13 | 己方后翼可易位 |
| 14 | 對方王翼可易位 |
| 15 | 對方后翼可易位 |
| 16 | 吃過路兵目標格（只有該格為 1） |
| 17 | 五十步計數 / 100.0（整層填同一個值） |

索引約定：`plane[rank][file]`，`rank=0` 是己方底線（a1 側，鏡射後）。

### 4.3 著法編碼：4672 維

採用 AlphaZero 的定義，`move_to_index(move) -> int`、`index_to_move(index, board) -> chess.Move`：

- 起始格 64 種 × 73 種「移動類型」
- 移動類型 0–55：**queen moves**，8 個方向 × 1–7 格距離
- 移動類型 56–63：**knight moves**，8 個方向
- 移動類型 64–72：**underpromotion**，3 個方向（直走、左吃、右吃）× 3 種棋子（N, B, R）
- 升變成后不另外編碼，走 queen moves 那一格即可

**必須實作雙向轉換並驗證 round-trip**：對隨機 10000 個盤面的所有合法著法做 `index_to_move(move_to_index(m), board) == m`。

`legal_mask(board) -> np.ndarray(4672, bool)` 也放這裡，推論時用來把非法著法的 logit 設成 `-inf`。

---

## 5. 前處理（`src/preprocess.py`）

PGN → 緊湊二進位檔。**不要把 `(18,8,8)` 的 float 存到硬碟**（那是 4.6 KB/盤面，會爆掉），改存壓縮表示，在 `Dataset.__getitem__` 才展開成張量。

### 每個盤面存 70 bytes

```
pieces      : int8[64]   # 0=空, 1..6=己方 PNBRQK, 7..12=對方 PNBRQK（已鏡射）
castling    : uint8      # 4 個 bit
ep_square   : int8       # -1 代表沒有
halfmove    : uint8
move_index  : uint16     # 0..4671，人類實際走的那步
result      : int8       # +1 己方贏, 0 和, -1 己方輸（從當前走棋方視角）
```

用 NumPy structured dtype 存成 `data/processed/train.npy` / `val.npy`，訓練時 `np.memmap` 讀取。

### 棋局篩選規則

- 兩方 Elo 都 ≥ 2000（`elite` 來源已滿足，仍要再檢查一次）
- 排除 bullet（`TimeControl` 起始秒數 < 180）
- 排除總步數 < 20 或 > 300 的棋局
- 排除 `Termination` 為 `Abandoned` / `Rules infraction` 的棋局
- 排除變體（`Variant` 存在且不是 `Standard`）

### 盤面取樣規則

- 跳過前 8 步（開局書階段，學了意義不大且會被記憶）
- 跳過最後 2 步
- 每盤棋最多取 40 個盤面（超過就均勻隨機抽），避免長棋被過度加權
- 和局的 `result` 為 0

### 切分

- 依**棋局**切分 train / val（不是依盤面！同一盤棋的盤面不可跨集合，否則洩漏）
- 比例 98 : 2
- 目標規模：train 約 **1500 萬–3000 萬盤面**（GPU 撐得住，資料量是這類模型棋力的主要瓶頸）。用 `--max-positions` 控制上限
- 單月 elite 檔約產出 1000 萬盤面，建議下載 2–3 個月合併；`preprocess.py` 要支援一次吃多個 PGN 檔（`--input data/raw/*.pgn`）

### 其他

- 用 `tqdm` 顯示進度，每 10000 局印一次統計（已處理局數、產生盤面數、被過濾原因分佈）
- 支援中斷續跑（分 shard 寫，已存在的 shard 跳過）

---

## 6. 模型（`src/model.py`）

```
輸入 (B, 18, 8, 8)
  ↓ Conv2d(18 → C, 3x3, padding=1) + BatchNorm + ReLU
  ↓ ResidualBlock × N
       每個 block: Conv(C→C,3x3) + BN + ReLU + Conv(C→C,3x3) + BN + 殘差相加 + ReLU
  ├─ Policy head:
  │    Conv2d(C → 73, 3x3, padding=1)   → flatten → (B, 4672) logits
  └─ Value head:
       Conv2d(C → 8, 1x1) + BN + ReLU → flatten(512) → Linear(512→256) + ReLU
       → Linear(256→1) → tanh → (B, 1)
```

`C = channels`、`N = blocks`，由 preset 決定（預設 `base`：C=128、N=10，約 7 M 參數）。
`forward()` 回傳 `(policy_logits, value)`。

補充要求：

- `blocks` 與 `channels` 從 `config.yaml` 讀，不要寫死
- 提供 `Model.from_checkpoint(path)` 類方法
- 提供 `count_parameters()` 並在訓練開始時印出來
- policy head 輸出的是 **raw logits**，softmax 與 legal mask 都在外面做

---

## 7. 訓練（`src/train.py`）

### 損失函數

```
loss = policy_loss + value_weight * value_loss
policy_loss = CrossEntropyLoss(policy_logits, move_index)      # 不做 legal mask，讓網路自己學會合法性
value_loss  = MSELoss(value.squeeze(), result.float())
value_weight = 1.0
```

### 超參數（寫進 `config.yaml`，這是預設值）

```yaml
device: auto
preset: base                 # small / base / large，見 §1

train:
  batch_size: 1024           # 由 preset 覆寫
  epochs: 12
  optimizer: adamw
  lr: 2.0e-3                 # batch 較大，lr 也跟著放大
  weight_decay: 1.0e-4
  scheduler: cosine          # warmup 1000 steps 後 cosine 降到 lr*0.05
  grad_clip: 1.0
  value_weight: 1.0
  amp: true                  # 自動選 bf16 / fp16
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  log_every: 100             # steps
  eval_every: 5000           # steps，跑 val 子集
  checkpoint_every: 1        # epochs
```

### 要求

- 每個 epoch 結束存 checkpoint 到 `models/epoch_{n}.pt`，並維護 `models/best.pt`（依 val policy top-1 準確率）
- checkpoint 內含 `model_state_dict`、`optimizer_state_dict`、`epoch`、`global_step`、`config`，可完整續訓（`--resume`）
- 訓練指標寫入 `logs/train_log.csv`：`step, epoch, lr, policy_loss, value_loss, policy_top1, policy_top5, value_mae`
- 用 `tqdm` 顯示 it/s 與 ETA
- **不要引入 wandb / tensorboard**，csv + 一支簡單的 matplotlib 畫圖腳本就好
- 支援 `--smoke-test`：只跑 200 步，用來確認整條路是通的

### 預期結果（寫進 README，讓使用者知道有沒有訓歪）

| 指標 | 1 epoch 後 | 12 epochs 後 |
|---|---|---|
| policy top-1 | ~35 % | 50–56 % |
| policy top-5 | ~68 % | 82–88 % |
| value MAE | ~0.72 | 0.55–0.62 |

top-1 若停在 10 % 以下，八成是著法編碼或鏡射寫錯了 → 回去跑 `tests/test_encoding.py`。
另外在 `train.py` 印出每秒處理的盤面數（positions/s）。若 GPU 使用率明顯偏低而 CPU 滿載，代表卡在 DataLoader，要調高 `num_workers` 或簡化 `__getitem__`。

---

## 8. 下棋（`src/search/greedy.py` + `src/play.py`）

Phase 1 的「搜尋」很簡單，但介面要為 Phase 2 準備好。

定義一個抽象介面（`src/search/__init__.py`）：

```python
class Searcher:
    def select_move(self, board: chess.Board) -> chess.Move: ...
    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]: ...
```

`GreedySearcher` 實作：

1. 編碼盤面（含鏡射）→ 前向傳播 → policy logits
2. 用 `legal_mask` 把非法著法設成 `-inf`
3. softmax
4. **鏡射還原**：如果原本輪到黑方，把 index 轉回來的 move 也要鏡射回原盤面座標
5. 依 `temperature` 參數決定：`temperature=0` 取 argmax，`>0` 則依機率抽樣

再加一層很便宜的補強（CPU 也負擔得起，且明顯提升棋力）：

- **一步將死檢查**：先掃所有合法著法，有立即將死就直接走
- **送子檢查**：對 policy 前 5 名的著法各走一步，用 value head 評估對手視角的分數，選對自己最好的那個（等同 depth-2 的極小化極大）。用 `config.yaml` 的 `search.top_k` 控制，預設 5

`src/play.py` 要提供兩種模式：

- `--mode cli`：終端機文字棋盤，人類輸入 UCI 著法（如 `e2e4`）與 AI 對弈
- `--mode uci`：實作最小可用的 UCI 協定（`uci` / `isready` / `position` / `go` / `quit`），這樣可以掛進 Arena、Cute Chess 等 GUI

---

## 9. 評估（`src/evaluate.py`）

三種評估，都要能單獨用參數呼叫：

1. `--mode accuracy`：在 val 集上算 policy top-1 / top-5、value MAE
2. `--mode match`：用 `python-chess` 的 `chess.engine.SimpleEngine` 開 Stockfish，打 N 局（預設 100，開局用隨機前 4 步製造變化，先後手各半）。對手強度用 `Skill Level`（0、3、5）或 `Limit(depth=1)`
3. `--mode baseline`：對「隨機合法著法」打 200 局，這是最低門檻，勝率應該 > 98 %

輸出勝/和/負與依標準公式估計的 Elo 差：`Elo_diff = -400 * log10(1/score - 1)`，`score = (勝 + 0.5*和) / 總局數`。同時輸出 ±95 % 信賴區間。

結果存成 `logs/eval_{timestamp}.json`。

---

## 10. 測試（`tests/`）

用 `pytest`。這些測試不是形式，是拿來救命的：

**`test_encoding.py`**
- 從 100 個隨機對局中取 10000 個盤面，對每個合法著法驗證 `index_to_move(move_to_index(m), board) == m`
- 驗證起始盤面的編碼：己方 8 個兵在 plane 0 的 rank 1、4 個易位權都是 1
- 驗證鏡射一致性：`encode(board)` 與 `encode(board.mirror())` 在對應位置上應完全相同（因為 canonical）
- 驗證 `legal_mask` 的 True 數量等於 `board.legal_moves.count()`

**`test_preprocess.py`**
- 用一個手寫的 3 局小 PGN 跑完整流程，驗證輸出筆數、`result` 的正負號方向正確（白方贏的棋局，白方走棋的盤面 result 應為 +1，黑方走棋的盤面應為 -1）

**`test_model.py`**
- 前向傳播形狀正確：`(4, 18, 8, 8)` → `(4, 4672)` 與 `(4, 1)`
- value 輸出落在 `[-1, 1]`
- `base` preset 的參數量落在 5 M–9 M 之間
- 一個 batch 的過擬合測試：對 8 筆資料訓練 200 步，loss 應降到 0.1 以下
- 所有測試都在 CPU 上跑（`device="cpu"`），確保程式碼沒有寫死 CUDA

---

## 11. Phase 2 預留（只寫介面與 TODO，不要實作）

`src/search/mcts.py` 建立骨架：

```python
class MCTSSearcher(Searcher):
    """AlphaZero 式 MCTS。Phase 2 實作。

    TODO:
      - Node: prior P, visit count N, value sum W, children
      - PUCT 選擇: Q + c_puct * P * sqrt(sum_N) / (1 + N)
      - 展開時用 model 一次前向取得 policy prior 與 value
      - 根節點加 Dirichlet noise (alpha=0.3, eps=0.25)
      - 回傳訪問次數分佈作為 policy target
    """
```

`src/selfplay.py` 建立骨架，註解說明 replay buffer 與訓練迴圈如何接回 `train.py`（`train.py` 的 dataset 介面要允許 policy target 是 4672 維機率向量，不只是整數 index — 請在 Phase 1 就把這個彈性留好，用 `soft_targets: bool` 開關）。

`config.yaml` 預留 `mcts:` 與 `selfplay:` 區塊，值先填預設但不使用。

---

## 12. 程式碼風格要求

- **註解用繁體中文**，技術名詞保留英文。每個函式都要有 docstring 說明輸入輸出的 shape 與意義
- 型別註解（type hints）全部加上
- **不要過度抽象**。這個專案的讀者剛開始學 class 與 OOP，能用函式解決的就不要開 class；真的要用 class 時，在檔案開頭用註解解釋這個 class 存在的理由
- 不要用 metaclass、decorator 魔法、動態 import
- 所有魔術數字都要有具名常數（例如 `NUM_MOVE_PLANES = 73`）
- 每支 `src/*.py` 都要能用 `python -m src.xxx --help` 單獨執行，用 `argparse`
- 錯誤訊息要能指出下一步該做什麼（例如找不到 `data/processed/train.npy` 時，直接印出應該跑哪一行指令）

---

## 13. 交付檢查清單

依序完成，每一步都要能實際跑起來再進下一步：

- [ ] **M0** `requirements.txt`、`config.yaml`、目錄結構、`README.md` 骨架
- [ ] **M1** `encoding.py` + `test_encoding.py`，`pytest tests/test_encoding.py` 全過
- [ ] **M2** `scripts/download_data.py`，能抓到 elite 單月檔
- [ ] **M3** `preprocess.py` + 測試，`--sample` 模式 5 分鐘內產出 `train.npy`
- [ ] **M4** `model.py` + `dataset.py` + 測試，overfit 8 筆資料成功
- [ ] **M5** `train.py --smoke-test` 200 步跑通，loss 有下降，且確認 `torch.cuda.is_available()` 為 True、GPU 有被用到
- [ ] **M6** 完整訓練跑完 1 個 epoch，val top-1 > 30 %
- [ ] **M7** `evaluate.py --mode baseline` 對隨機走法勝率 > 98 %
- [ ] **M8** `play.py --mode cli` 可以人機對弈
- [ ] **M9** `evaluate.py --mode match` 對 Stockfish Skill Level 0 有勝場
- [ ] **M10** `search/mcts.py`、`selfplay.py` 骨架與 TODO 就位

每完成一個里程碑，在終端機印出該跑的驗收指令，等確認通過再繼續。
