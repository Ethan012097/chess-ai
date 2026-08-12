# 西洋棋 AI（監督式學習，GPU）

用人類高分棋局做監督式學習，訓練一個「雙頭神經網路」：

- **Policy head**：給定盤面，預測人類會走哪一步（4672 維）
- **Value head**：給定盤面，預測這盤棋最後誰會贏（-1 ~ +1）

然後用這個網路直接下棋。Phase 2 會把同一個網路接上 MCTS + 自我對弈，
所以 Phase 1 的所有設計（canonical 視角、4672 維著法編碼、policy logits + value 輸出）
都已經是 Phase 2 相容的。

---

## 0. 安裝

### 0.1 建立虛擬環境（Python 3.12）

```powershell
cd C:\code\chess_ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 0.2 安裝 PyTorch（CUDA 版）

**這一行一定要單獨跑，而且要帶 `--index-url`**，否則會裝到 CPU 版：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 0.3 自檢：確認 GPU 有被認到

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

預期輸出類似：

```
2.13.0+cu126 True NVIDIA GeForce RTX 3060 Laptop GPU
```

**如果 `cuda.is_available()` 是 `False`**，多半是裝到 CPU 版 wheel。解法：

```powershell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

（`pip list` 裡如果看到 `torch  2.x.x` 而不是 `torch  2.x.x+cu126`，就是裝錯了。）

### 0.4 安裝其餘套件

```powershell
pip install -r requirements.txt
```

`requirements.txt` **刻意不含 torch**，避免蓋掉上面裝好的 CUDA 版。

### 0.5 Stockfish（只有評估時需要）

不會自動下載。請到 <https://stockfishchess.org/download/> 抓 Windows 版，
解壓後把執行檔放到 `bin/stockfish.exe`（路徑寫在 `config.yaml` 的 `eval.stockfish_path`）。

沒裝 Stockfish 也能跑 `--mode baseline`（對隨機走法）與 `--mode accuracy`。

---

## 1. 選 preset（依你的 VRAM）

| preset | 適用 VRAM | channels | blocks | batch_size | 實際參數量 |
|---|---|---|---|---|---|
| `small` | 4–6 GB | 96 | 8 | 512 | 1,541,434 |
| `base` | 8–12 GB（預設） | 128 | 10 | 1024 | 3,192,026 |
| `large` | 16 GB 以上 | 192 | 14 | 1536 | 9,591,322 |

用 `--preset small` 之類的參數切換，或改 `config.yaml` 的 `preset:`。

查看目前生效的設定與參數量：

```powershell
python -m src.config --preset small
python -m src.model --preset small
```

> **關於參數量**：CLAUDE.md §1 的表格寫的是 3.5M / 7M / 22M，但那組數字跟同一份文件
> §6 定義的網路架構對不起來——照 §6 的結構算，`channels=128, blocks=10` 就是 3.19M，
> 三個 preset 都差了約 2.2 倍。本專案以「§6 的架構 + §1 的 channels/blocks」為準
> （這兩者互相吻合）。若你要的是宣稱的參數量，把 `config.yaml` 裡三個 preset 的
> `blocks` 分別改成 20 / 23 / 33 即可，其他程式碼完全不用動。

**遇到 `CUDA out of memory`**：換小一號的 preset，或把 batch_size 減半：

```powershell
python -m src.train --preset small
python -m src.train --batch-size 256
```

---

## 2. 逐步執行

### 步驟 1：下載棋譜

兩個來源都是直接下載，**不需要註冊、也不需要 API key**。

```powershell
# 預設來源 elite（單月約 77 MB 壓縮 / 234 MB 解壓）
python scripts/download_data.py --source elite --month 2025-11

# 只想快速跑通整條 pipeline（前 5000 局）
python scripts/download_data.py --source elite --month 2025-11 --sample

# 看有哪些來源、目前最新月份、以及 Phase 1.5 的備案
python scripts/download_data.py --list
```

| source | 壓縮格式 | 說明 |
|---|---|---|
| `elite` | `.zip` | **預設。** Lichess Elite Database，第三方對官方資料做二次篩選：只留 **2500+ 分對上 2300+ 分**、且已排除 bullet。單月約 77 MB，30 幾萬局 |
| `lichess` | `.pgn.zst` | Lichess 官方開放資料庫，**CC0 授權**，可自由下載 / 修改 / 再散布。沒篩選過，什麼分數都有，單月光壓縮檔就約 28 GB。只有想自訂篩選條件時才用 |
| `ccrl` | `.7z` | 引擎對局，PGN 內含每步評分，Phase 2 可當 value 的額外標註（標準庫解不開 `.7z`，需手動解壓） |
| `pgnmentor` | `.zip` | 大師棋譜，檔案小、乾淨，適合快速煙霧測試 |

**兩個主要來源的壓縮格式不同，解壓路徑也不同**：`elite` 走 `zipfile`，
`lichess` 走 `zstandard` 的 stream reader（邊解壓邊寫，不會把 28 GB 整包攤到硬碟）。
下載完會依副檔名自動分流。

下載會顯示進度條，檔案已存在就跳過（要重抓加 `--force`）。

> **月份要挑存在的。** `database.nikonoel.fr` 對不存在的月份會回 **HTTP 200 + HTML 錯誤頁**
> 而不是 404，所以下載腳本會檢查檔頭的魔術位元組（`.zip` 要以 `PK\x03\x04` 開頭），
> 抓到網頁就直接報錯並提示換月份。實測 **2025-11 是目前最新可用的月份**，
> 2025-12 之後都是錯誤頁。

**資料量**：單月 elite 實測產出約 **1100 萬個訓練盤面**。規格的目標是 1500 萬–3000 萬，
所以建議抓 **2–3 個月**再一起前處理（`preprocess.py` 支援一次吃多個檔案）：

```powershell
python scripts/download_data.py --source elite --month 2025-11
python scripts/download_data.py --source elite --month 2025-10
python scripts/download_data.py --source elite --month 2025-09
python -m src.preprocess --input "data/raw/*.pgn"
```

預期輸出：

```
[下載] https://database.nikonoel.fr/lichess_elite_2024-01.zip
elite_2024-01.zip: 100%|██████████| 68.1M/68.1M [00:11<00:00, 6.02MB/s]
[完成] data\raw\lichess_elite_2024-01.pgn（234.1 MB）
```

### 步驟 2：前處理（PGN → .npy）

```powershell
python -m src.preprocess --input "data/raw/*.pgn"

# 限制盤面數（目標規模 1500 萬–3000 萬）
python -m src.preprocess --input "data/raw/*.pgn" --max-positions 20000000

# 快速跑通用
python -m src.preprocess --input "data/raw/*.pgn" --max-games 3000
```

**路徑一定要加引號**，PowerShell 不會自己展開 `*`。

每個盤面只存 70 bytes（不是 4.6 KB 的 float 張量），所以 2000 萬盤面約 1.3 GB。
支援中斷續跑：分 shard 寫，重跑同一個指令會跳過已完成的檔案。

> ⚠️ **同一個輸出資料夾一次只能跑一個 preprocess。** 合併階段會把 shard 資料夾裡
> **所有** shard 併進去然後刪掉，所以兩個行程同時寫 `data/processed` 會互相吃掉
> 對方的資料（而且切出來的 train / val 可能含有同一盤棋，造成洩漏）。
> 想在完整前處理跑到一半時另外做小規模測試，請指定不同的資料夾：
>
> ```powershell
> python -m src.preprocess --input "data/raw/*.pgn" --max-games 3000 --output-dir data/processed_test
> ```

輸出長這樣（實際數字依你抓的月份而定）：

```
讀取棋局 XXX,XXX，保留 XXX,XXX
總共過濾 X,XXX 局： 步數<20=..., 步數>300=..., Elo<2000=...

完成：
  data\processed\train.npy  XX,XXX,XXX 盤面（XXXX.X MB）
  data\processed\val.npy       XXX,XXX 盤面（XX.X MB）
```

參考速度（本機 elite 2024-01 實測）：約 145 局/秒、5,600 盤面/秒（單執行緒，CPU-bound）。

檢查產出：

```powershell
python -m src.dataset --split train
python -m src.dataset --split train --benchmark
```

### 步驟 3：煙霧測試（確認整條路是通的）

```powershell
python -m src.train --smoke-test --preset small
```

只跑 200 步。**重點是看 `policy_loss` 有沒有從 ~8.4（= ln 4672，亂猜的水準）往下掉。**

預期輸出：

```
======================================================================
preset        : small（channels=96, blocks=8）
參數量        : 1,541,434
裝置          : cuda（NVIDIA GeForce RTX 3060 Laptop GPU, 6.0 GB）
train 盤面    : 102,400
batch_size    : 512
======================================================================
[AMP] 使用 bfloat16（不需要 GradScaler）
epoch 1/1: 100%|██████████| 200/200 [00:15<00:00, p_loss=3.867, top1=14.6%, pos/s=14,339]

--smoke-test 完成（200 步）
  val top1=17.40% top5=44.19% value_mae=0.6145
```

### 步驟 4：完整訓練

```powershell
python -m src.train --preset small

# 續訓
python -m src.train --preset small --resume models/epoch_3.pt
```

- 每個 epoch 結束存 `models/epoch_{n}.pt`，並依 val top-1 維護 `models/best.pt`
- 訓練指標寫進 `logs/train_log.csv`
- checkpoint 內含 optimizer / scheduler 狀態，可完整續訓

畫訓練曲線：

```powershell
python scripts/plot_training.py
```

### 步驟 5：評估

```powershell
# val 集準確率
python -m src.evaluate --mode accuracy

# 對隨機走法打 200 局（最低門檻，勝率應 > 98%）
python -m src.evaluate --mode baseline

# 對 Stockfish 打 100 局（需要 bin/stockfish.exe）
python -m src.evaluate --mode match --skill-levels 0 3 5
```

結果會存成 `logs/eval_{timestamp}.json`，並輸出 Elo 差與 95% 信賴區間。

### 步驟 6：下棋

```powershell
# 終端機對弈
python -m src.play --mode cli

# 人類執黑（AI 先走）
python -m src.play --mode cli --black

# UCI 模式，可以掛進 Arena / Cute Chess
python -m src.play --mode uci
```

CLI 模式的著法可以用 **SAN**（`e4`、`Nf3`、`O-O`）或 **UCI**（`e2e4`），
輸入非法著法時會把所有合法著法列出來。指令：

| 指令 | 作用 |
|---|---|
| `undo` | 悔棋兩步（人與 AI 各一步，回到還是你走的狀態） |
| `hint` | 看 AI 覺得最好的前 5 步 |
| `fen` | 印出目前盤面的 FEN |
| `save` | 存成 PGN 到 `logs/games/`，可貼到 lichess |
| `flip` | 換邊 |
| `quit` | 離開（結束時會自動存 PGN） |

每步之後會顯示 AI 的信心、評估與耗時：

```
AI: c5   (信心 10%, 評估 +0.01, 0.45s)
候選: Nf6 18% | d5 15% | e6 15% | d6 14% | c5 10%
```

（信心低於候選第一名是正常的——送子檢查會用 value head 覆寫 policy 的首選。）

---

## 3. 預期結果（用來判斷有沒有訓歪）

CLAUDE.md §7 給的目標值：

| 指標 | 1 epoch 後 | 12 epochs 後 |
|---|---|---|
| policy top-1 | ~35 % | 50–56 % |
| policy top-5 | ~68 % | 82–88 % |
| value MAE | ~0.72 | 0.55–0.62 |

### 本機實測（`small` preset、1113 萬盤面、val 集 226,958 個盤面）

| epoch | policy top-1 | policy top-5 | policy loss | value MAE |
|---|---|---|---|---|
| 1 | 45.90 % | 85.11 % | 1.7138 | 0.7516 |
| 2 | 48.06 % | 86.75 % | 1.6259 | 0.7465 |
| 5 | 50.14 % | 88.39 % | 1.5392 | 0.7361 |
| 8 | 51.19 % | 88.92 % | 1.5051 | 0.7300 |

**policy 兩個指標都比目標值好**：第 1 個 epoch 的 top-1 就有 45.9 %（目標 ~35 %），
top-5 85.1 %（目標 ~68 %），到第 8 個 epoch 已經進入「12 epochs 後」的目標區間。

**value MAE 明顯達不到目標，而且幾乎不再下降**：8 個 epoch 只從 0.752 走到 0.730，
照這個斜率跑滿 12 個 epoch 也到不了 0.55–0.62。這**不是** bug，是 value 標籤本身的
雜訊上限造成的：

- value target 只有 -1 / 0 / +1 三種值，而且是「這盤棋最後誰贏」
- 中局盤面的勝負本來就有真實的不確定性，優勢方也可能後來超時或失誤輸掉，
  那個明明是好棋的盤面就被標成 -1
- 全部猜 0 的話 MAE 大約是 0.9（等於決勝局的比例），所以 0.73 確實有學到東西，
  只是這個標籤能提供的資訊就到這裡

想真的壓下 value MAE，要換更乾淨的標籤，見 [§7 Phase 1.5](#7-phase-15-備案更好的-value-標籤目前不實作)。
**在 policy 指標正常的前提下，value MAE 卡在 0.7 附近是預期行為，不用回頭找 bug。**

**top-1 若停在 10 % 以下**，八成是著法編碼或鏡射寫錯了：

```powershell
python -m pytest tests/test_encoding.py
```

**GPU 使用率明顯偏低而 CPU 滿載**，代表卡在 DataLoader（資料供給端跟不上）。
訓練時會印出 `pos/s`（每秒處理的盤面數），可以拿來判斷：

- 調高 `config.yaml` 的 `train.num_workers`
- 或簡化 `dataset.py` 的 `__getitem__`

參考數字：RTX 3060 Laptop (6GB) + `small` preset 約 14,000 盤面/秒。

---

## 4. 測試

```powershell
python -m pytest                        # 全部
python -m pytest tests/test_encoding.py # 編碼（最重要）
```

全部 173 條測試都在 CPU 上跑，不需要 GPU，也不需要訓練好的 checkpoint
（用隨機權重的小模型即可，搜尋與編碼的正確性跟棋力無關）。

| 測試檔 | 守住什麼 |
|---|---|
| `test_encoding.py` | 4672 維著法編碼的 round-trip、canonical 鏡射一致性、legal mask |
| `test_preprocess.py` | `result` 的正負號方向、篩選規則、train/val 依棋局切分 |
| `test_model.py` | 前向傳播形狀、value 值域、參數量、**8 筆資料的過擬合測試** |
| `test_pgn_writer.py` | `[%eval]` 的白方視角轉換、PGN 標頭 |
| `test_uci.py` | UCI 協定的四個坑、**可宣告和棋不能回 `bestmove 0000`** |
| `test_evaluate.py` | Wilson 信賴區間、SPRT、cutechess 輸出解析、**棄權局警告**、Spearman |
| `test_mcts.py` | **Q 值視角**、PUCT 用 `-child.q()`、終局偵測、時間管理 |
| `test_web.py` | **評估條視角**（鏡射盤面）、API 欄位、非法著法不改變盤面 |
| `test_selfplay.py` | 稀疏 policy round-trip、soft target 損失公式、**value target 逐步變號** |

`test_encoding.py` 的 round-trip 測試會對隨機盤面的每個合法著法驗證
`index_to_move(move_to_index(m), board) == m`——這是整個專案最容易寫錯、
錯了又最難發現的地方（loss 照樣會下降，只是永遠學不好）。

---

## 5. 目錄結構

```
chess_ai/
├── CLAUDE.md               # 專案規格書
├── README.md
├── requirements.txt        # 不含 torch，見 §0.2
├── config.yaml             # 所有超參數集中在這
├── pytest.ini
├── conftest.py
├── data/
│   ├── raw/                # 下載的 .pgn（不進 git）
│   └── processed/          # 前處理後的 .npy（不進 git）
├── models/                 # checkpoint（不進 git）
├── logs/                   # 訓練曲線 csv、評估結果 json
├── bin/                    # stockfish.exe 放這裡
├── src/
│   ├── config.py           # 讀 config.yaml → dataclass
│   ├── encoding.py         # 盤面編碼、著法編碼（Phase 1/2 共用）
│   ├── preprocess.py       # PGN → .npy shards
│   ├── dataset.py          # torch Dataset / DataLoader
│   ├── model.py            # 雙頭網路
│   ├── train.py            # 訓練迴圈
│   ├── evaluate.py         # 準確率 + 對局測試 + Elo 估計
│   ├── play.py             # CLI 對弈 / UCI 介面
│   ├── move_info.py        # MoveInfo + value ↔ centipawn 換算
│   ├── pgn_writer.py       # PGN 輸出（含 lichess 的 [%eval] 註解）
│   ├── selfplay.py         # 自我對弈迴圈（§6.9）
│   ├── search/
│   │   ├── greedy.py       # policy + 一步將死檢查 + 送子檢查
│   │   └── mcts.py         # AlphaZero 式 MCTS
│   └── web/
│       ├── server.py       # FastAPI 後端（§6.8）
│       └── static/
│           └── index.html  # 全部前端程式碼就這一個檔
├── scripts/
│   ├── download_data.py    # 下載 + 解壓資料
│   ├── make_demo_games.py  # 產生示範對局 PGN
│   ├── make_openings.py    # 產生開局書
│   └── plot_training.py    # 畫訓練曲線
└── tests/
```

每支 `src/*.py` 都可以單獨執行看說明：

```powershell
python -m src.encoding --help
python -m src.model --preset base
python -m src.dataset --split train
```

---

## 6. 設計重點

### Canonical orientation（正規化視角）

**永遠從「輪到走棋的一方」的視角編碼。** 輪到黑方時先把棋盤上下鏡射並交換顏色
（`board.mirror()`），使「自己」永遠是白方、永遠往上前進。

這樣網路只要學一種視角，資料效率加倍，Phase 2 的 MCTS 也能直接沿用。

### 輸入張量 `(18, 8, 8)`

| planes | 內容 |
|---|---|
| 0–5 | 己方的 P, N, B, R, Q, K |
| 6–11 | 對方的 P, N, B, R, Q, K |
| 12–15 | 己方王翼 / 己方后翼 / 對方王翼 / 對方后翼 易位權（整層填 0 或 1） |
| 16 | 吃過路兵目標格（只有該格為 1） |
| 17 | 五十步計數 / 100.0（整層填同一個值） |

索引約定：`plane[rank][file]`，`rank=0` 是己方底線。

### 著法編碼 4672 維

64 個起始格 × 73 種移動類型：

- 0–55：queen moves，8 方向 × 1–7 格
- 56–63：knight moves，8 方向
- 64–72：underpromotion，3 方向 × 3 種棋子（N, B, R）
- 升變成后不另外編碼，走 queen moves 那一格即可

### 儲存格式（每個盤面 70 bytes）

```
pieces      : int8[64]   # 0=空, 1..6=己方 PNBRQK, 7..12=對方 PNBRQK（已鏡射）
castling    : uint8      # 4 個 bit
ep_square   : int8       # -1 代表沒有
halfmove    : uint8
move_index  : uint16     # 0..4671
result      : int8       # +1 己方贏, 0 和, -1 己方輸（當前走棋方視角）
```

訓練時用 `np.load(path, mmap_mode="r")` 讀（回傳的就是 `np.memmap`），
在 `Dataset.__getitem__` 才用純 NumPy 展開成 `(18,8,8)` 張量。

### Phase 1 的「搜尋」

1. policy logits → legal mask（非法設 -inf）→ softmax → 鏡射還原
2. **一步將死檢查**：有立即將死就直接走
3. **送子檢查**：對 policy 前 5 名各走一步，用 value head 評估對手視角的分數，
   選對自己最好的（等同 depth-2 極小化極大）

用 `config.yaml` 的 `search.top_k`、`search.use_mate_check`、`search.use_lookahead` 控制。

---

## 6.4 UCI 引擎與 Cute Chess（Phase 2 第 2 節）

實作完這一節，這個專案就是一個標準西洋棋引擎，可以掛進任何 GUI。

### 快速自檢（不用裝 GUI）

```powershell
# 直接跟引擎講 UCI
"uci`nisready`nposition startpos`ngo movetime 100`nquit" | .\.venv\Scripts\python.exe -m src.play --mode uci

# 或透過 GUI 會用的那個包裝檔
"uci`nisready`nposition startpos`ngo movetime 100`nquit" | cmd /c engine.bat

# 自動化測試（12 條，不需要開 GUI）
python -m pytest tests/test_uci.py -v
```

預期輸出：

```
id name ChessAI-SL
id author chess-ai (Phase 1, supervised)
option name Temperature type spin default 0 min 0 max 100
option name TopK type spin default 5 min 1 max 20
option name Checkpoint type string default models/best.pt
option name MCTS type check default false
option name Simulations type spin default 800 min 1 max 100000
uciok
readyok
info depth 1 seldepth 2 score cp 249 nodes 20 nps 107 time 186 pv d2d4
bestmove d2d4
```

支援的指令：`uci`、`isready`、`ucinewgame`、`position startpos|fen ... [moves ...]`、
`go`（可解析 `movetime` / `wtime` / `btime` / `winc` / `binc` / `depth` / `nodes` /
`infinite`）、`stop`、`setoption`、`quit`。

可設定的 option（GUI 裡直接調）：

| option | 型別 | 說明 |
|---|---|---|
| `Temperature` | spin 0–100 | **百分比**（UCI 的 spin 只能是整數）。30 = temperature 0.3 |
| `TopK` | spin 1–20 | 送子檢查要看幾個候選 |
| `Checkpoint` | string | 換模型，會即時重新載入 |
| `MCTS` | check | Phase 2 預留，目前設了不會有作用 |
| `Simulations` | spin | 同上 |

### Cute Chess 設定步驟

1. 到 <https://cutechess.com/> 下載 Windows 版並安裝
2. **Tools → Settings → Engines → Add**
   - Name：`MyNet`
   - Command：專案根目錄的 `engine.bat`
   - Working Directory：`C:\code\chess_ai`
   - Protocol：`UCI`
3. 同樣方式再加一個 `Stockfish`，Command 指向 `bin\stockfish.exe`
4. 按 OK 時 Cute Chess 會實際啟動引擎確認 `uciok` ——
   **這一步失敗就代表 UCI 有問題**，先看 stderr

三種看它下棋的方式：

- **人機對弈**：Game → New，Player 1 選 Human、Player 2 選 MyNet
- **引擎對打**（推薦，可以坐著看）：Player 1 選 MyNet、Player 2 選 Stockfish
  並設 `Skill Level=0`，時控 10 秒 + 0.1 秒增秒
- **批次對局**：Tournament → New，在背景跑幾百局

### 疑難排解

| 症狀 | 多半的原因 |
|---|---|
| **引擎載入失敗** | stdout 被污染，或 stdout 緩衝沒開。把 `engine.bat` 最後一行改成帶 `2> "%~dp0engine_err.log"` 的版本（檔案裡有註解寫好），再看那個 log |
| **下到一半引擎消失** | 通常是吐出非法著法被判負。檢查 stderr；`_choose_move` 有防護，理論上不會發生 |
| **反應很慢** | 確認模型只在啟動時載入一次，不是每次 `go` 都重載。stderr 應該只有開頭出現一次 `[uci] 已載入 ...` |
| **中文變亂碼** | `src/__init__.py` 已統一把 stdout/stderr 切成 UTF-8，不要繞過它 |

### UCI 的四個坑（都已處理，改程式時別退回去）

1. **stdout 緩衝** — Python 接到 pipe 時是塊緩衝，`print` 的字可能根本沒送出去，
   GUI 等到超時就判引擎當機。`UciEngine.run()` 開頭有 `reconfigure(line_buffering=True)`
2. **stdout 只能有 UCI 訊息** — 任何除錯輸出、tqdm、PyTorch 警告跑到 stdout 都會
   讓 GUI 解析失敗。除錯一律走 `log_stderr()`，且 `play.py` 在 import torch 前就
   `warnings.filterwarnings("ignore")`
3. **絕不吐出非法著法** — GUI 收到非法著法直接判負且不說原因。`_choose_move()`
   一定檢查 `move in board.legal_moves`，任何例外都退回第一個合法著法
4. **升變與易位格式** — 升變寫 `e7e8q`、易位寫王的起訖格 `e1g1`（不是 `e1h1`）。
   全部交給 python-chess 的 `Move.uci()` / `push_uci()`，不要自己拼字串

## 6.45 引擎對抗與 Elo 評估（Phase 2 第 3 節）

### 安裝 cutechess-cli

免安裝可攜版，同一包裡 GUI 與 CLI 都有：

```powershell
$ver = "1.5.1"
$url = "https://github.com/cutechess/cutechess/releases/download/v$ver/cutechess-$ver-win64.zip"
Invoke-WebRequest $url -OutFile "$env:TEMP\cutechess.zip"
Expand-Archive "$env:TEMP\cutechess.zip" -DestinationPath "bin\" -Force

# 驗證
& "bin\cutechess-1.5.1-win64\cutechess-cli.exe" --version
```

路徑寫在 `config.yaml` 的 `eval.cutechess_path`。缺 DLL 時執行那包附的
`vc_redist.x64.exe`。

### 開局書

自動對戰若每局都從起始盤面開始，同一個確定性引擎會下出一模一樣的棋，統計毫無意義。

```powershell
python scripts/make_openings.py --input "data/raw/*.pgn"
python scripts/make_openings.py --input "data/raw/*.pgn" --min-count 20 --plies 6
```

取每盤棋的前 8 個半步、要求該開局在原始資料至少出現 50 次（避免冷僻變化），
去重後隨機取 2000 個，輸出 `data/openings.pgn`。

> **資料量與門檻的取捨**：`--min-count 50` 這個門檻相當嚴。單月 elite（約 30 萬局）
> 只有 **743** 個開局過關；要湊到 2000 個需要 2–3 個月的資料。
> 想用單月資料就湊到 2000，把門檻降到 `--min-count 20`（單月可得 1,833 個）。
> 腳本跑完會告訴你有幾個過關，不足時也會提示怎麼調。

### 對戰與 Elo

```powershell
# 跟 Stockfish Skill 0 打 100 輪（= 200 局，先後手各半）
python -m src.evaluate --mode tournament --skill-level 0 --rounds 100

# 判斷新的 checkpoint 有沒有比舊的強（SPRT 序貫檢定）
python -m src.evaluate --mode sprt --new models/epoch_12.pt --old models/best.pt
```

`--mode tournament` 會組出 cutechess-cli 的指令並解析它印出的結果。
幾個參數的意義（`config.yaml` 的 `eval:` 區塊可調）：

| 參數 | 意義 |
|---|---|
| `-games 2 -rounds N -repeat` | 每輪打兩局，**同一開局位置先後手各一次**。`-repeat` 不能省，否則白方優勢會污染 Elo |
| `-concurrency` | 同時跑幾局。**每個引擎實例都會載入一份模型到 GPU**，設太高會 OOM，預設 2 |
| `tc=10+0.1` | 每方 10 秒 + 每步 0.1 秒增秒 |

**SPRT 是判斷「新版有沒有比較強」的正確工具。** 它邊打邊做序貫檢定，一旦統計上
有結論就自動停止，通常幾百局就夠，不必固定打滿。預設檢定「是否強 20 Elo 以上」。

> **沒有 SPRT 就不要相信自己的直覺。** 100 局的勝率差在 ±5 % 以內幾乎沒有統計
> 意義，但人眼看起來會覺得「新版明顯比較強」。

### Lichess 謎題測驗（把 top-1 翻譯成人話）

```powershell
python -m src.evaluate --mode puzzles --checkpoint models/best.pt
python -m src.evaluate --mode puzzles --puzzle-count 200      # 快速版
```

從 <https://database.lichess.org/lichess_db_puzzle.csv.zst> **串流**讀題
（整包 304 MB，但各分桶收滿就停，實際只會下載前面幾 MB），依 Rating 分桶
（`<1000`、`1000-1200`、…、`2000+`）輸出各桶命中率。

產出的數字（例如「1000 分以下答對 78 %、1800 分答對 12 %」）是最能對外說明成果的
指標，也是之後驗證 MCTS 有沒有用的**對照組**——同一批題目接上搜尋再跑一次。

> **一個很容易搞錯的地方**：CSV 裡的 `FEN` 是**對手走之前**的盤面，
> 要先 push `Moves` 的第一步才得到真正要解的局面。直接拿 FEN 去解會整批答錯，
> 而且因為每題看起來都「有解出一步」，錯得很難察覺。

**心理準備**：純 policy 網路在深度戰術上會很難看，四步組合殺基本上是猜。
這正是 MCTS 要解決的問題。

#### 本機實測基準（`models/best.pt`，2000 題）

| Rating 分桶 | 命中率 | 答對/嘗試 |
|---|---|---|
| `<1000` | **87.7 %** | 250/285 |
| `1000-1200` | 73.7 % | 210/285 |
| `1200-1400` | 69.5 % | 198/285 |
| `1400-1600` | 60.4 % | 172/285 |
| `1600-1800` | 56.8 % | 162/285 |
| `1800-2000` | 51.6 % | 147/285 |
| `2000+` | **47.4 %** | 135/285 |
| **總計** | **63.9 %** | 1274/1995 |

命中率隨難度單調下降，從 87.7 % 掉到 47.4 % —— 這正是「有直覺、但看不深」的
典型曲線。**把這張表存起來**，之後 MCTS 做完要拿同一批題目重跑對照，
高分桶的提升幅度就是搜尋的價值。

（附帶一提，這裡的成績有一部分要歸功於 `GreedySearcher` 的一步將死檢查與送子
檢查——低分桶的題目很多就是「將死一步」或「吃免費子」，那兩層便宜的補強直接
處理掉了。純 policy argmax 的分數會更低。）

### Elo 信賴區間為什麼用 Wilson

`--mode match` / `--mode baseline` 自己算 Elo 時，用的是**得分率的 Wilson 區間**，
再把點估計與上下界各自代進同一個 `elo_difference`。

課本上的 Wald 區間（`p ± z·sqrt(p(1-p)/n)`）在這裡會壞掉：得分率逼近 0 或 1 時
變異數塌成 0，區間縮成一個點，**點估計反而跑到區間外面**。M7 的 198勝2和0負
就踩到了，印出 `+920（CI: +768 ~ +800）` 這種矛盾輸出。

換成 Wilson 之後：

| 情境 | Wald（壞） | Wilson（對） |
|---|---|---|
| 198勝2和0負 | +920（CI +768 ~ +800）← 點估計在區間外 | +920（CI +618 ~ ≥+1200） |
| 200 局全和 | +0（CI +0 ~ +0）← 假的精確 | +0（CI −48 ~ +48） |
| 2 局全勝 | 區間退化 | ≥+1200（CI −113 ~ ≥+1200）← 正確表達「2 局證明不了什麼」 |

`tests/test_evaluate.py` 用 `assert lower <= point <= upper` 把這件事釘死了。

## 6.5 PGN 呈現層（Phase 2 第 1 節）

所有對局一律存成標準 PGN，附上 **lichess 認得的評分註解**。

```powershell
# 自我對弈 10 局，存成 logs/demo/game_{n}.pgn
python scripts/make_demo_games.py

python scripts/make_demo_games.py --games 3 --temperature 0.5
python -m src.pgn_writer --demo          # 只想看註解格式長什麼樣
```

每一步會寫成這樣：

```
1. e4 { [%eval 0.36] policy: e4 .41 d4 .22 Nf3 .11 c4 .08 e3 .04 | 0.03s }
```

- `[%eval x.xx]` 是 lichess 的標準格式，單位是兵值、**白方視角**。
  把 PGN 貼到 <https://lichess.org/paste> 就會自動畫出評分曲線
- 後半段是自己的除錯資訊，lichess 會忽略

`--temperature` 預設 0.3 而非 0：0 是純 argmax，模型是確定性的，10 局會下出
一模一樣的棋。

### 視角約定（整個專案錯誤率最高的地方）

| 場合 | 視角 |
|---|---|
| `MoveInfo.value`、value head 輸出、訓練資料的 `result` | **當前走棋方** |
| PGN 的 `[%eval]`、網頁評估條 | **白方** |
| UCI 的 `score cp` | **當前走棋方** |

只有寫進 PGN 與網頁時要轉成白方視角（`MoveInfo.value_white()`），其餘一律維持
走棋方視角。`tests/test_pgn_writer.py` 把這件事釘死了。

> **關於 eval 數值偏大**：cp 換算用的是 Leela 公式
> `290.68 * tan(1.548 * value)`，在 value 接近 ±1 時會急遽放大
> （value 0.9 → 15.9 兵）。所以評分曲線看起來會比 Stockfish 誇張很多，
> 尤其 value head 本身還很吵（見 §3）。這是顯示問題不影響棋力，
> 想要溫和一點可以改用 `value_to_cp(v, linear=True)`（`600 * value`）。

## 6.6 value head 品質（Phase 2 第 5 節）

```powershell
python -m src.evaluate --mode value-quality --checkpoint models/best.pt
python -m src.evaluate --mode value-quality --value-positions 500 --value-depth 10
```

從 val 集抽 3000 個盤面，用 Stockfish depth 12 各評一次，計算 **Spearman 等級
相關係數**與**正負號一致率**。

### 為什麼不看 MAE

`CLAUDE.md` §7 寫的「value MAE 應降到 0.55–0.62」**是錯的**。MAE 對「最終誰贏」
這種標籤沒有有意義的下界：假設某盤面真實勝率是 0.6，完美的評估器會輸出 +0.2，
但標籤只會是 +1 或 −1，誤差恆為 0.8 或 1.2。**就算把 Stockfish 本人當 value head，
對這個 val 集算出的 MAE 也不會低到 0.5。**

MCTS 真正需要的是**盤面排序能力**（哪個局面比較好），不是絕對數值準不準
——所以量 Spearman，不量 MAE。

### 本機實測結果

| 指標 | 數值 |
|---|---|
| Spearman 等級相關 | **0.7642** |
| 正負號一致率 | 76.2 % |
| 樣本 | 3,000 個盤面，Stockfish depth 12 |

對照兩條門檻：

- 進入第 6 節 MCTS 的門檻是 **0.60** → 通過
- 第 5 節「用 Stockfish 評分重訓 value」的驗收目標是 **0.75** → **已經達到**

**結論：value 重訓（§5.4 / §5.5）目前不需要做。** 那要下載 20 GB 的評分資料庫
再重訓好幾個小時，而它的驗收標準我們沒做就已經達標了。

這也解釋了先前 value MAE 卡在 0.73 的現象：**MAE 難看不代表 value 沒用**，
它的排序能力其實不錯，只是被「最終誰贏」這個三值標籤的雜訊蓋住了。

> 依規格 §6.6，如果 MCTS 做完只贏 greedy 五六十 Elo（而不是預期的 200+），
> **value 品質是第一嫌疑犯**，那時候再回頭做第 5 節的重訓。

## 6.7 MCTS（Phase 2 第 6 節）

```powershell
# 對單一盤面跑一次搜尋，看訪問次數分佈
python -m src.search.mcts --simulations 800
python -m src.search.mcts --fen "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1" --simulations 100

# 在 UCI 裡開啟（Cute Chess 的 option 也可以設）
setoption name MCTS value true
setoption name Simulations value 800

# 對打 greedy，用 SPRT 判定強多少
python -m src.evaluate --mode sprt --mcts --simulations 800 --rounds 100
```

### 視角約定（這裡最容易寫錯）

**每個 Node 的 Q 與 `value_sum`，都是從「該節點輪到走棋的那一方」的視角。**
由此推出兩件事，兩件都很容易寫反：

1. **選擇子節點時要用 `-child.q()`** —— 子節點的 Q 是對手視角，父節點看到的好壞
   要取負號。搞反的話 AI 會積極送子，因為它以為對手變好是好事
2. **回溯時每往上一層 value 取一次負號** —— 雙人零和，我的 +1 就是你的 −1

`tests/test_mcts.py` 有專門的視角測試（`test_root_q_sign_matches_network_value`、
`test_backup_alternates_sign`、`test_puct_uses_negated_child_q`）把這兩件事釘死。

### GPU 批次化

單執行緒 MCTS 每次模擬只做 batch=1 的前向，GPU 使用率會掉到個位數。
所以用 **virtual loss + 批次葉節點評估**：一次選 32 條路徑到葉節點，
每選中一個就先加 virtual loss（假裝輸了）避免整批擠在同一條路上，
32 個盤面一起前向，再展開回溯並扣除 virtual loss。

實測 **約 1,500 次模擬/秒**（RTX 3060 Laptop，`small` preset），800 次模擬約 0.53 秒。

### 實測結果

**對打 greedy（SPRT，10+0.1 時控）**

```
Score of MCTS800 vs greedy: 64 - 16 - 7  [0.773] 87
Elo difference: 215.7 +/- 85.4, LOS: 100.0 %
SPRT: llr 2.98, H1 was accepted
結束原因：46 White mates / 34 Black mates / 7 三次重複和 / 1 No result
```

**+215.7 ± 85.4 Elo**，87 局就達到統計顯著，達成規格 §6.6 的「強 200 Elo 以上」。
（`No result` 那一局是 SPRT 判定後另一條並行的對局被中止，不是錯誤。）

> **這個數字是修正 §6.9 那個棄權 bug 之後重跑的。** 修正前的結果是
> `80-26-1，+193.0 ± 77.9`—— 結論相同，但當時 107 局裡只有 1 局和棋（0.9 %），
> 因為所有三次重複都被判成了棄權負；修正後和局率回到正常的 8 %。
> 兩次的信賴區間互相涵蓋，所以**正確的說法是「大約 200 Elo，誤差 ±100」**，
> 不是「從 193 進步到 215」。

**謎題測驗（同一批題目，各 100 題/桶）**

| Rating | greedy | MCTS400 | 差 |
|---|---|---|---|
| `<1000` | 89.0 % | 82.0 % | −7.0 |
| `1000-1200` | 76.0 % | 78.0 % | +2.0 |
| `1200-1400` | 73.0 % | 66.0 % | −7.0 |
| `1400-1600` | 65.0 % | 59.0 % | −6.0 |
| `1600-1800` | 62.0 % | 62.0 % | 0.0 |
| `1800-2000` | 45.0 % | **58.0 %** | **+13.0** |
| `2000+` | 45.0 % | **58.0 %** | **+13.0** |
| 總計 | 65.0 % | 66.1 % | +1.1 |

**兩個最難的分桶各提升 13 個百分點**，這正是規格預期的「高分桶大幅提升」。
中低分桶小幅下降：每桶只有 100 題，標準誤約 5 %，−6 ~ −7 大約是 1.2σ，
落在雜訊範圍內；而 +13 約 2.5σ，是真的。

低分桶沒有進步也說得通 —— 那些題目多半是「一步將死」或「吃免費子」，
`GreedySearcher` 的一步將死檢查與送子檢查本來就直接處理掉了，搜尋沒有補上空間。
**真正的強度證據是 SPRT 的 +193 Elo**，謎題只是輔助說明搜尋補在哪裡。

（上表的 MCTS 只跑 400 次模擬以節省時間；正式對局用 800 次會更好。）

---

## 6.8 網頁棋盤與策略箭頭（Phase 2 第 4 節）

棋盤上疊著箭頭，粗細代表模型對每個候選著法的信心，左邊一條評估條顯示局勢。

```powershell
# 啟動（模型只在啟動時載入一次）
python -m src.web.server

# 或指定 checkpoint / 用 MCTS 當箭頭來源
python -m src.web.server --checkpoint models/epoch_12.pt --mcts --simulations 400

# 也可以直接用 uvicorn（設定從環境變數讀）
uvicorn src.web.server:app --host 127.0.0.1 --port 8000
```

然後開瀏覽器到 <http://127.0.0.1:8000>。同源，不需要 CORS 設定。

**第一次開需要能連上外網**：`chessboard.js`（顯示）、`chess.js`（合法性）、jQuery
與棋子圖片都從 CDN 載入。這是刻意的 —— 沒有 npm、沒有 webpack、沒有建置流程，
三個月後想重跑只要 `python -m src.web.server` 一行。CDN 載不到時頁面上方會出現紅色橫幅說明。

### 三種模式

| 模式 | 用途 |
|---|---|
| **對弈** | 你拖曳走子，AI 回應。AI 落子前會先讓箭頭停留 800 ms，讓你看見它在考慮什麼 |
| **自動播放** | AI 自我對弈，每步間隔可調（預設 1500 ms），可暫停、可單步。**最適合展示** |
| **分析** | 貼上任意 FEN，只顯示箭頭與評估，不走棋 |

網址可以直接指定模式：`http://127.0.0.1:8000/?mode=auto&autostart=1`
開一個分頁就自己開始自我對弈，不用點任何按鈕。

### 箭頭怎麼畫

- 線寬 `4 + 20 * sqrt(prob)`。**用 sqrt 不用線性** —— 線性的話 3 % 的著法會細到看不見，
  但那些正是你想觀察的
- 透明度 `0.35 + 0.55 * prob`，顏色是同一色相的深淺變化（機率越高越深）。
  用不同顏色會讓人誤以為代表不同類別
- 機率 < 3 % 就不標百分比，避免擠成一團；最佳著法額外加粗一級
- 箭頭數量用滑桿調 1–8。後端一律回傳 8 個，所以拉滑桿是即時重畫，不會重新請求

### policy 箭頭 vs MCTS 箭頭

右上角的標題會標明目前顯示的是哪一種：

- **Policy 機率** —— 「直覺想走哪裡」
- **MCTS 訪問次數** —— 「想過之後認為哪裡值得」

**這兩者意義不同，不能混為一談。** 在「搜尋方式」勾選「使用 MCTS」再按套用就會切換
（模型不會重載，只換搜尋器）。並排比較是很好的展示點：可以直接看出搜尋修正了直覺的哪些地方。

候選清單下方若出現綠色的「實際走：X（送子檢查否決了 Y）」，代表 policy 最想走的那步
被 depth-2 的送子檢查擋下來了 —— 那正是 `GreedySearcher` 兩層補強發揮作用的時候。

### 為什麼切到 MCTS 之後評估條會跳動

policy 模式的評估條用 value head 對當前盤面的**直接輸出**；MCTS 模式用**根節點的 Q**
（`root.q()`，跟 UCI 的 `score cp` 同一個來源）。兩者是不同的估計，數字不會一樣。

差最多的是**起始盤面**：

| 盤面 | value head 直接輸出 | MCTS 400 `root.q()` | MCTS 3200 `root.q()` |
|---|---|---|---|
| 起始盤面 | **+0.458**（+2.49 兵） | −0.127 | +0.014 |
| 白方多一后 | +0.794 | +0.790 | +0.791 |
| 黑方多一后 | −0.867 | −0.796 | −0.801 |

**子力失衡的盤面兩者幾乎完全一致**（這也是視角沒寫反最有力的證據）；
只有起始盤面差很多，而且模擬次數拉高就收斂回 0 附近。

原因是 value head 對起始盤面**過度自信**：它給白方 +2.49 兵，但只要白方走完任何一步，
下一個盤面的評估就掉回 0.0 附近（實測 1.d4 之後是 +0.02 兵）。這個 +0.458 是訓練資料裡
起始盤面出現次數極多、而白方勝率確實略高所造成的記憶，不是真的評估。
**MCTS 往前看一層就把它修正掉了** —— 這正好是「搜尋修正直覺」最好的示範。

### API

| 端點 | 說明 |
|---|---|
| `POST /api/analyse` | `{fen, top_k, temperature}` → value、候選著法、AI 實際會走的 `best` |
| `POST /api/move` | `{fen, uci}` → 新的 FEN；非法時回 `legal:false` 與**原本的** FEN |
| `POST /api/pgn` | 整局的著法紀錄 → 含 `[%eval]` 註解的 PGN（重用 `pgn_writer.py`） |
| `POST /api/config` | 切換 policy / MCTS |
| `GET /api/health` | 目前載入的 checkpoint、裝置、參數量 |

**視角處理全部集中在後端**（`server.value_payload()` 是唯一做轉換的地方），
前端只認 `value_white`，不做任何加減號。這是避免評估條畫反最有效的方法。

推論端點寫成同步 `def` 而不是 `async def`：FastAPI 看到同步函式會自動丟到 threadpool，
不會阻塞事件迴圈；寫成 `async def` 反而會把整個伺服器卡住。

### 驗收實測

```powershell
python -m pytest tests/test_web.py -q      # 24 條，全部在 CPU 上跑
```

| §4.5 驗收項 | 結果 |
|---|---|
| 起始盤面最粗的箭頭是常見開局著法 | Nf3 39.0 %（實際走 d4，送子檢查改的） |
| 白方優勢盤面評估條白色過半，黑方相反 | 多一后 → 89.7 % / 10.3 %；多一車 → 86.3 % |
| 拖曳非法著法棋子彈回，盤面不會壞掉 | chess.js 立即 snapback，後端再確認一次 |
| 自動播放跑完一整局並正確顯示結果 | 90 手，`0-1（將死）` |
| 匯出的 PGN 貼到 lichess 能正常載入 | 90 步全部可重播，90 個 `[%eval]` 註解 |

視角測試在 `tests/test_web.py` 裡是這樣做的：因為盤面編碼是 canonical 的，
**同一個盤面與它的鏡射餵進網路會得到完全相同的 value**，於是「走棋方視角」應該相同、
「白方視角」應該正負相反。少取或多取一次負號這條就會爆。

---

## 6.9 自我對弈訓練迴圈（Phase 2 第 7 節）

```powershell
# 先跑煙霧測試（4 局 + 50 步，約 30 秒，不會動到 models/best.pt）
python -m src.selfplay --smoke-test

# 一個完整 iteration
python -m src.selfplay --games 500 --train-steps 2000

# 連續跑 5 代
python -m src.selfplay --iterations 5
```

### 先講清楚期望值

AlphaZero 從零開始用了 5000 顆 TPU，**你有一張顯卡**。所以這裡不是從零開始，
而是從監督式模型出發繼續進化。合理預期：跑一到兩週，棋力提升幾十到一兩百 Elo，
然後進入緩慢爬升。若跑了三天完全沒進步，先回頭跑 `pytest tests/test_mcts.py`
檢查視角，而不是加大模型。

### 一個 iteration 的四個步驟

1. **產生對局** —— 用目前的 `best.pt` 自我對弈，每步 400 次模擬、根節點加 Dirichlet noise
2. **寫入 replay buffer** —— 稀疏格式存成 `data/selfplay/iter_XXXX.npy`
3. **訓練** —— 從 buffer 抽樣訓練 M 步，`soft_targets: true`
4. **把關** —— SPRT 對打舊版，**只有 SPRT 判定 H1 才更新 `best.pt`**

第 4 步是防止「訓練損失下降但棋力變差」的唯一防線。SPRT 沒有結論時一律當作沒變強
—— 寧可多跑一代，也不要讓一個沒被證明比較強的模型污染後續所有自我對弈資料。
`--no-gate` 會跳過這一步，只在除錯時用。

### 稀疏 policy 儲存格式（§7.2）

MCTS 的 policy target 是 4672 維分佈，存成 float32 是 18 KB/盤面，
100 萬盤面就 18 GB。但訪問次數本來就極度集中，取前 32 名就夠：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `pieces` | `int8[64]` | 同 Phase 1 |
| `castling` / `ep_square` / `halfmove` | `uint8` / `int8` / `uint8` | 同 Phase 1 |
| `top_indices` | `uint16[32]` | 訪問次數前 32 名的著法 index |
| `top_probs` | `uint16[32]` | 機率 × 65535 |
| `value_target` | `int16` | 值 × 10000（當前走棋方視角） |

**197 bytes/盤面**，比 dense 省 95 倍。載入時展開成 4672 維、未列入處補 0 並重新正規化。

前處理時會統計「被截斷掉的機率總和」，平均超過 0.02 就代表 `SPARSE_POLICY_K` 太小。
實測是 **0.0000** —— 400 次模擬下平均只有個位數個著法被訪問過，32 個名額綽綽有餘。

### 兩個 Phase 1 預留的擴充點怎麼打開的

**`soft_targets`**：規格要求的公式是 `-(target * log_softmax(logits)).sum(dim=1).mean()`。
我們沒有另外寫一個分支 —— `F.cross_entropy` 吃機率向量時算的就是這條公式，
而且是融合過的實作，數值更穩。`tests/test_selfplay.py` 直接比對兩者的數值來證明，
不是「應該一樣」而是真的量過。

**`ChessPositionDataset`**：靠 dtype 自動判斷讀到的是監督式（70 bytes）還是
自我對弈（197 bytes）資料，`train.py` 一行都不用改。自我對弈資料配
`soft_targets: false` 會當場報錯，不會默默取 argmax 把 MCTS 想過的東西丟掉。

### 認輸機制（§7.5）

根節點 value 連續 10 步低於 `selfplay.resign_threshold`（預設 −0.9）就判負，
省下大量無意義的殘局步數。**保留 10 % 的對局不啟用認輸**用來統計誤判率：
那些局照樣下完，事後檢查「本來會認輸的那一方最後是不是真的輸了」。
誤判率應低於 5 %，超過就把門檻調嚴（−0.9 → −0.95）。

### best.pt 與 candidate.pt 是兩個不同的東西

這是整個迴圈最容易寫錯的地方：

| 檔案 | 用途 | 什麼時候更新 |
|---|---|---|
| `models/best.pt` | **產生對局**用 | 只有 SPRT 判定 H1 才更新 |
| `models/selfplay_candidate.pt` | **持續訓練**用 | 每代都在它上面繼續練（含 Adam 動量） |

直覺會想每代都從 `best.pt` 開始訓練，但那樣的話：把關沒過 → 訓練成果全丟掉 →
下一代從同一個起點、用差不多的資料再練一次 → 結果也差不多 → **永遠過不了關**。
把「產生資料的模型」與「正在訓練的模型」分開才會累積。`--from-best` 可以強制重來。

跑迴圈之前建議先留一份監督式模型的副本，之後才比得出來：

```powershell
Copy-Item models\best.pt models\best_supervised.pt
```

### 健康指標（§7.6）

每代寫進 `logs/selfplay_log.csv`，用這支畫成圖：

```powershell
python scripts/plot_selfplay.py      # → logs/selfplay_curves.png
```

**和局比例是最重要的一項**：超過 70 % 代表模型過度保守
（常見於 value 主導、policy 多樣性不足），要調高 `mcts.dirichlet_epsilon`
或 `mcts.temperature_moves`。程式在超標時會直接把該調哪個參數印出來。

### 第一個 iteration 的實測結果

```
60 局 → 2,773 盤面，平均 46.2 手，和局 3.3 %，截斷機率質量 0.0000
1000 步訓練：policy_loss 2.42, value_loss 0.093
SPRT（500 局）：201 勝 201 敗 98 和，+0.0 ± 27.3 Elo
→ 沒有通過把關，保留舊的 best.pt ✓
```

**候選模型跟原本的一模一樣強** —— 500 局打下來 Elo 差是 0.0，
95 % 信賴區間 ±27.3 已經窄到足以說「真實差距小於 27 Elo」。

這是正確也是預期的結果，不是 bug：2,773 個盤面被 1000 步 × batch 256 抽了 92 遍，
那不是學習而是背誦（value_loss 掉到 0.093 就是過擬合的徵兆，
監督式階段的 value MAE 是 0.73）。要有實質進步得靠正式規模：
500 局/代 × 20 代滑動視窗，累積到幾十萬盤面。

順帶一提，SPRT 在這裡**沒有給出 H0/H1 結論**（llr −1.03，門檻 ±2.94）。
真實差距剛好是 0 的時候，llr 爬得很慢 —— 要接受 H0 大概得打一千多局。
遇到這種情況不必硬跑：`±27.3 Elo` 本身就已經回答了問題。

### 要跑多久（實測）

400 次模擬、平均 48 手的一局大約 **12–25 秒**（單行程，RTX 顯卡）。所以：

| 設定 | 產生對局 | SPRT 把關 | 一代合計 |
|---|---|---|---|
| 60 局（試跑） | 約 20 分 | 約 30 分 | 約 50 分 |
| 500 局（正式） | 約 3 小時 | 約 30 分 | 約 3.5 小時 |

**P21 的「連續 5 代」大約要 18 小時**，是掛著跑一晚上的事情，不是坐在旁邊等的事情。
瓶頸是 CPU（python-chess 的著法產生），不是顯卡 —— 見下面那個效能陷阱。
真的要加速應該開多行程，`config.yaml` 的 `selfplay.num_workers` 是為此預留的。

### 一個一直在偷輸棋的 bug（已修）

跑第一個 iteration 的 SPRT 時，對打紀錄裡出現這個：

```
Finished game 77 (candidate vs best): 0-1 {White makes an illegal move: 0000}
```

**80 局裡有 28 局是這樣結束的 —— 35 %。**

原因是 `play.py` 的 `handle_go` 寫成：

```python
if self.board.is_game_over(claim_draw=True):    # ← 錯的
    print("bestmove 0000")
```

三次重複與五十步規則是「**可以宣告**」而不是「自動成立」，盤面上還有合法著法。
引擎回了 `0000`（UCI 的空著法），Cute Chess 判定為 illegal move 直接判負。
**和棋要交給 GUI 裁決，引擎該做的是繼續走棋。**

三個地方都有同一份錯誤，全部改成判斷「有沒有合法著法」：
`play.py` 的 `handle_go`、`GreedySearcher.select_move`、`MCTSSearcher.select_move`。

修好之後同樣的對打：**0 局棄權**，而且原本那些棄權局變成了正確的
`{Draw by 3-fold repetition}`。

**哪些數字受影響？只有經過 cutechess-cli 的那些。** 分界線在「誰當裁判」：

| 評估模式 | 誰在跑對局 | 受影響嗎 |
|---|---|---|
| `--mode baseline`（對隨機） | 程式內部直接呼叫 `Searcher` | **否** |
| `--mode match`（對 Stockfish） | 程式內部 + `chess.engine.SimpleEngine` | **否** |
| `--mode tournament` | cutechess-cli ← 走 UCI | **是** |
| `--mode sprt` | cutechess-cli ← 走 UCI | **是** |

前兩種的對局迴圈自己用 `is_game_over(claim_draw=True)` 判定並記成和局，
兩邊一視同仁，那是合理的裁決；只有走 UCI 的那條路會把 `0000` 當成非法著法。
所以 M7 的 198勝2和0負 與 M9 對 Stockfish 的結果**不受影響**，
只有 §6.7 的 MCTS vs greedy 需要重跑。那次 107 局只有 1 局和棋，
現在回頭看就是徵兆：重複和棋全都被算成勝負了。

**已重跑，結論不變**：`64-16-7，+215.7 ± 85.4 Elo，H1 accepted`，零筆棄權，
和局率回到 8 %。MCTS 的優勢是真的，沒有被 bug 灌水。

`tests/test_uci.py` 補了兩條會抓到它的測試（五十步、三次重複）。

教訓：**引擎對打的紀錄要看結束原因，不要只看比分。** 比分是 39-39 看起來很正常，
結束原因才顯示出三分之一的局數根本沒在下棋。所以 `evaluate.py` 的
`--mode tournament` 與 `--mode sprt` 現在一律列出結束原因分佈，
出現非法著法棄權時會直接警告「這是程式 bug 不是棋力問題，這場結果不能採信」：

```
  結束原因：
     36 局（ 60.0 %）Black mates
     12 局（ 20.0 %）White mates
     12 局（ 20.0 %）Draw by 3-fold repetition
```

### 一個效能陷阱（已修）

第一版 MCTS 在選擇階段每經過一個節點就呼叫一次 `legal_indices()`，
而那是純 Python 的著法產生 + 4672 維編碼。實測 400 次模擬花 331 ms，
其中 GPU 只佔 25 ms —— **九成以上的時間卡在 Python，不是顯卡**。

把對照表在展開時記進 `Node.moves`（位置固定，對照表就固定）之後降到 **143 ms**，
快 2.3 倍。順帶一提，batch size 從 32 加大到 128 反而更慢：virtual loss 會讓
同一批的路徑大量重疊，多做的樹走訪抵銷掉 GPU 那點好處。**32 是實測的甜蜜點。**

這也說明為什麼規格 §7.5 建議的「跨局合併 batch」在這個專案幫助有限：
GPU 本來就閒著。要再加速應該用多行程（真正的 CPU 平行），而不是更大的 batch。

## 7. Phase 1.5 備案：更好的 value 標籤（目前不實作）

Phase 1 的 value target 是「這盤棋最後誰贏」。這個標籤其實**很吵**：優勢方可能後來
超時輸掉，那個明明是好棋的盤面就被標成 -1。所以 value MAE 有個天花板，
單靠加資料量或加訓練時間是壓不下去的。

更乾淨的做法是改用 Stockfish 的評分當 value target：

- <https://database.lichess.org/lichess_db_eval.jsonl.zst>（約 20 GB）
- Lichess 的 Stockfish 評分資料庫，約 **3.9 億個**已評估盤面，
  由使用者在瀏覽器裡跑 Lichess 分析板產生的
- 接法：把 centipawn 分數用 `tanh(cp / 400)` 之類的方式壓到 [-1, 1]，
  取代 `POSITION_DTYPE` 的 `result` 欄位即可，其餘程式碼完全不用改

**判斷什麼時候該換**：如果 value MAE 卡住怎麼訓都下不去，而 policy top-1 還在正常
上升，那就是 value 標籤的雜訊到頂了，這時候換才划算。

**本專案已經觀察到這個現象**（見 §3 實測表）：8 個 epoch 內 policy top-1 從 45.9 %
爬到 51.2 %，但 value MAE 只從 0.752 挪到 0.730。也就是說網路還在學，
只是 value 那一側已經逼近「最終誰贏」這個標籤的資訊上限了。

不過先把優先順序講清楚：**Phase 1 的下棋強度主要靠 policy head**
（`GreedySearcher` 用 policy 選點、value 只做一層送子檢查），policy 指標是好的，
所以現在換 value 標籤的邊際效益不高。真正會讓 value 變關鍵的是 Phase 2 的 MCTS
——那時候每個葉節點都靠 value 評估，雜訊會直接變成棋力損失。
**建議的順序是：先做 Phase 2 的 MCTS，遇到 value 拖後腿時再回頭換這個資料集。**

---

## 8. Phase 2 預留

Phase 2 的預留項目**都已經打開了**：

- `src/search/mcts.py`：完成（§6.7）—— PUCT、Dirichlet noise、訪問次數分佈、GPU 批次化
- `src/web/`：完成（§6.8）—— 網頁棋盤與策略箭頭
- `src/selfplay.py`：完成（§6.9）—— 對局產生、replay buffer、訓練、SPRT 把關
- `train.soft_targets`：dataset 回傳 4672 維機率向量而不是整數 index，
  MCTS 的訪問次數分佈可以直接當 policy target，`train.py` 一行都不用改
- `config.yaml` 的 `mcts:` 與 `selfplay:` 區塊現在都真的會被讀取

`data/selfplay/` 也不進 git（跟 `data/` 其餘部分一樣）。
