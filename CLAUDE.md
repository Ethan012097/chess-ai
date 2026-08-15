# CLAUDE.md — 西洋棋 AI 專案規格與進度

這份文件是本專案的**唯一規格書**（原 CLAUDE.md 與 CLAUDE-PHASE2.md 已整合於此）。
詳細的實測數據、踩過的坑與操作步驟寫在 `README.md`，這裡只放**規格、約定、與目前進度**。

---

## 0. 目前進度（2026-08-16）

### 已完成

| 階段 | 內容 | 狀態 |
|---|---|---|
| **Phase 1** | M0–M10 監督式學習全流程 | ✅ 全部完成 |
| **Phase 2** | P1–P20（PGN、UCI、Elo 評估、網頁、MCTS、自我對弈） | ✅ 全部完成 |
| **P21** | 連續 5 代自我對弈 | ✅ 跑完了，但**沒有任何一代通過把關** |

**198 條測試全過**（`python -m pytest tests/ -q`），全部在 CPU 上跑，不需要 GPU 或 checkpoint。

### P21 的結果

| 代 | 盤面 | 平均步數 | 和局 | value_loss | Elo（vs best） | SPRT |
|---|---|---|---|---|---|---|
| 1 | 17,760 | 88.8 | 33 % | 0.255 | 0.0 | `?` |
| 2 | — | — | — | — | — | 中斷，資料留著但從未被訓練 |
| 3 | 17,714 | 88.6 | 31.5 % | 0.377 | 0.0 ± 55.4（LOS 50 %） | `?` |
| 4 | 17,039 | 85.2 | 29.5 % | 0.356 | +20.3 ± 54.0（LOS 77 %） | `?` |
| 5 | 18,019 | 90.1 | 32.5 % | 0.322 | **−67.4** ± 57.2（LOS 1 %） | `?` |

`models/best.pt` 全程沒有被換掉（還是 8/12 的監督式模型），把關規則有正常執行。
資料在 `data/selfplay/iter_0001..0005.npy`，統計在 `logs/selfplay_log.csv`，
曲線 `logs/selfplay_curves.png`。

**兩個結論要記住：**

1. **這個 SPRT 設定沒有解析度，三個 `?` 幾乎不帶資訊。**
   `--gate-rounds 60` = 最多 120 局，而量到的信賴區間是 ± 55 Elo。
   要在 `elo0=0 / elo1=20` 之間分辨勝負需要上千局，llr 只走到 −0.257（邊界 ±2.94）。
   `?` 不是「沒有進步」的證據，是「這把尺量不出來」。
2. **第 5 代是真的變差了**（LOS 1.0 %、llr −1.91 往 H0 走）。
   第 3 代從 `best.pt` 重新開始，第 4、5 代才接續候選模型 ——
   也就是「持續訓練」一啟動，兩代之後掉了 67 Elo。
   每代只有 200 局新資料卻練 1500 步 @ lr 1e-4，像是把監督式學到的東西洗掉了。

另外 **認輸率四代都是 0.0 %**：−0.9 連 10 步的門檻一次都沒觸發，
認輸機制等於沒作用，每代白算了不少必敗殘局。

### 怎麼啟動長時間的跑（重要）

**用 `scripts/run_selfplay_detached.ps1`，不要用 `Start-Process`。**

```powershell
# 接續跑 3 代（代數自己從 buffer 的檔案數接下去），順便擋掉睡眠
powershell -ExecutionPolicy Bypass -File scripts\run_selfplay_detached.ps1 -Iterations 3 -BlockSleep

# 監看
Get-ScheduledTaskInfo -TaskName chessai_selfplay
Get-Content logs\selfplay_<時間戳>.log -Tail 20        # 健康指標 / SPRT
dir data\selfplay          # 每 25 局更新一次，超過 20 分鐘沒動才是真的停了

# 停止 / 收尾
Stop-ScheduledTask -TaskName chessai_selfplay
python scripts/plot_selfplay.py
powercfg /change standby-timeout-ac 10   # 把睡眠設定改回來
```

理由：`Start-Process` 啟動的行程仍在「啟動它的那個 session 的行程樹」底下，
終端機關閉或 session 結束都可能連坐。排程器啟動的任務父行程是 Task Scheduler 服務，
不受影響（見 §5 的 13）。每代約 75 分鐘（含 SPRT），3 代約 4 小時。

**中斷的代價很小**：每 25 局落地（`SAVE_EVERY_GAMES`）、每代一行 csv、
候選模型含 Adam 動量存在磁碟上、代數由 buffer 檔案數推導 ——
續跑就是重下同一道指令。唯一的缺口是中斷在一代中途時，
那個只寫了一半的 shard 會被算成完整的一代（第 2 代就是這樣）。

### 模型現況

| 檔案 | 內容 | 說明 |
|---|---|---|
| `models/best.pt` | small，C96×8，1.54 M 參數 | **目前使用中** |
| `models/best_supervised.pt` | 同上的備份 | 位元組相同，另有 GitHub v1.0 Release |
| `models/base/best.pt` | base，C128×10，3.19 M 參數 | 訓練完成但 **SPRT 未通過，未採用** |
| `models/epoch_*.pt` | small 的逐 epoch checkpoint | 保留供比較 |

### 關鍵實測數字

```
val top-1 51.60% / top-5 89.13% / value MAE 0.7253   （small，12 epoch，1113 萬盤面）
對隨機走法          198-0-2      99.0%
對 Stockfish Skill 0 18-9-3      65.0%（30 局，CI 跨 0，樣本不足）
MCTS800 vs greedy   27-0-3       +512 Elo [+251, +772]
value 品質          Spearman 0.7642 / 正負號一致 76.2%（對 Stockfish depth 12）
```

### 下一步

優先序是**先修把關的解析度，再調訓練配方** —— 尺不準的時候調參數只是在猜。

1. **放寬 SPRT 的對立假設**（`elo0=0 / elo1=50`），120 局就有機會出結論。
   代價是只抓得到「明顯變強」。改一行、零額外成本。
2. **查第 5 代到底壞在哪**：`--mode puzzles --compare-with`（McNemar 配對檢定，已實作），
   看是戰術能力退化還是別的。不用重跑自我對弈，半小時有答案。
3. **調訓練配方**：每代 500 局、訓練步數降到 500–800，減少在少量資料上過擬合。
   代價是每代拉長到 3 小時以上。
4. 讓認輸機制真的會觸發（目前四代都是 0 %），省下必敗殘局的算力。
5. 若要量絕對棋力：`--mode match --uci-elo 1320 1400 1500`（已實作，需 150+ 局才有意義）
6. Phase 1.5 備案（用 Stockfish 評分重訓 value）目前**不需要** —— Spearman 0.7642 已超過 0.75 的驗收目標

---

## 1. 專案目標

用人類高分棋局做監督式學習訓練「雙頭神經網路」（policy + value），
再接上 MCTS 與自我對弈，成為 AlphaZero 簡化版。

環境：Windows 11 + PowerShell、Python 3.12 venv、RTX 3060 Laptop 6 GB。
PyTorch 用 CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu126`

---

## 2. 不可更動的核心約定

改這些就會全面壞掉，而且**大多不會報錯**，只會讓棋力悄悄變差。

### 2.1 視角（本專案錯誤率最高的地方）

**約定：value 一律是「當前走棋方」的視角。** 由此推出：

| 場合 | 視角 | 轉換 |
|---|---|---|
| value head 輸出、`MoveInfo.value`、UCI `score cp` | 走棋方 | 不用轉 |
| PGN 的 `[%eval]`、網頁評估條 | **白方** | 黑方走棋時取負號 |
| MCTS 的 `Node.q()` / `value_sum` | 該節點走棋方 | 父節點看子節點要用 `-child.q()` |
| `_backup` 回溯 | 每往上一層 | 取一次負號 |

網頁的視角轉換**只在 `server.value_payload()` 一處**發生，前端只認 `value_white`。

### 2.2 盤面與著法編碼

- **Canonical orientation**：永遠鏡射成「輪到走的一方在下方」（`board.mirror()`）
- 輸入張量 `(18, 8, 8)`：0–5 己方 PNBRQK、6–11 對方、12–15 易位權、16 吃過路兵、17 五十步/100
- 著法編碼 **4672 維**（64 格 × 73 planes：56 queen + 8 knight + 9 underpromotion）
- 儲存格式：監督式 **70 bytes**/盤面（`POSITION_DTYPE`），自我對弈 **197 bytes**（`SELFPLAY_DTYPE`，稀疏前 32 名）

### 2.3 把關規則

- **`models/best.pt` 只有 SPRT 判定 H1 才能覆蓋。** 沒結論一律當作沒變強
- 換 preset 重訓**務必加 `--model-dir`**，否則會蓋掉舊 checkpoint 且繞過把關
- 自我對弈的「產生對局模型」（best.pt）與「持續訓練模型」（selfplay_candidate.pt）**必須分開**，
  否則把關沒過就永遠原地踏步

---

## 3. 目錄結構

```
chess_ai/
├── CLAUDE.md              # 本檔（規格 + 進度）
├── README.md              # 實測數據、踩過的坑、操作步驟
├── config.yaml            # 所有超參數
├── data/  models/  logs/  bin/     # 都不進 git
├── src/
│   ├── config.py  encoding.py  preprocess.py  dataset.py  model.py  train.py
│   ├── evaluate.py        # accuracy / match / baseline / tournament / sprt / puzzles / value-quality
│   ├── play.py            # CLI 對弈 + UCI 引擎
│   ├── move_info.py  pgn_writer.py  selfplay.py
│   ├── search/            # greedy.py（policy + 將死檢查 + 送子檢查）、mcts.py
│   └── web/               # server.py（FastAPI）+ static/index.html（全部前端）
├── scripts/               # download_data / make_openings / make_demo_games / plot_* /
│                          # check_opening_value / run_selfplay_detached.ps1（長時間跑用這支）
└── tests/                 # 8 個測試檔，198 條
```

每支 `src/*.py` 都能 `python -m src.xxx --help` 單獨執行。

---

## 4. 各模組要點

### 4.1 前處理（`preprocess.py`）

篩選：兩方 Elo ≥ 2000、排除 bullet（< 180 秒）、步數 20–300、排除 Abandoned/變體。
取樣：**跳過前 8 個半步**（`range(8, ...)`，所以 ply 0–7 從未進過訓練集）、跳過最後 2 步、
每盤最多 40 個盤面。依**棋局**切分 train/val（98:2），不是依盤面。

### 4.2 模型（`model.py`）

`Conv(18→C) → ResBlock × N → policy head (C→73, flatten 4672) + value head (C→8→256→1, tanh)`

| preset | channels | blocks | batch | 參數量 |
|---|---|---|---|---|
| small | 96 | 8 | 512 | 1.54 M |
| base | 128 | 10 | 1024（6 GB 用 512） | 3.19 M |
| large | 192 | 14 | 1536 | 9.59 M |

### 4.3 訓練（`train.py`）

`loss = policy_loss + value_weight * value_loss`

`F.cross_entropy` 同時吃整數 index（監督式）與機率向量（自我對弈的 `soft_targets`）——
後者算的就是 `-(target * log_softmax(logits)).sum(dim=1).mean()`，不需要另寫分支。

### 4.4 搜尋

**`GreedySearcher`**：policy argmax + 一步將死檢查 + 送子檢查（前 k 名各走一步用 value head 評估）。

**`MCTSSearcher`**：PUCT + Dirichlet noise + virtual loss + GPU 批次化（batch 32 是實測甜蜜點）。
根節點也有一步將死檢查，與 greedy 共用 `find_mate_in_one()`。

### 4.5 評估（`evaluate.py`）

| mode | 用途 | 走 cutechess？ |
|---|---|---|
| `accuracy` | val 集 top-1/top-5/MAE | 否 |
| `baseline` | 對隨機走法 | 否 |
| `match` | 對 Stockfish（`--skill-levels` 或 `--uci-elo`） | 否 |
| `tournament` / `sprt` | cutechess-cli 對打、SPRT 序貫檢定 | **是** |
| `puzzles` | lichess 謎題，`--compare-with` 做 McNemar 配對檢定 | 否 |
| `value-quality` | 對 Stockfish 評分算 Spearman | 否 |

Elo 信賴區間用 **Wilson**（不是 Wald），不變式 `lower <= point <= upper` 有 assert 守著。

### 4.6 自我對弈（`selfplay.py`）

一代四步：產生對局 → 寫入 replay buffer → 抽樣訓練 → **SPRT 把關**。
replay buffer 保留最近 20 代、抽樣偏向新資料。認輸門檻 −0.9 連續 10 步，保留 10 % 審計局統計誤判率。

---

## 5. 已知陷阱（都踩過，都有測試守著）

改動相關程式碼前先看這一節。**這些 bug 的共同點是表面指標完全正常。**

1. **引擎在可宣告和棋時回 `bestmove 0000`** → Cute Chess 判為非法著法直接判負。
   80 局裡輸掉 28 局，比分卻是 39-39 看起來很正常。
   判斷依據必須是「有沒有合法著法」，不是 `is_game_over(claim_draw=True)`。

2. **MCTS virtual loss 符號寫反** → 剛選過的路徑變成分數最高，整批葉節點擠到同一個子節點。
   症狀是訪問次數出現「剛好等於 batch size 的整齊倍數」。
   `value_sum` 是節點自己的視角，要讓父節點覺得沒吸引力必須 `+=` 而不是 `-=`。

3. **未訪問節點的 Q 用 0** → 劣勢局面（已探索的 −Q ≈ −0.9）裡每個沒走過的著法都顯得更好，
   搜尋攤平、prior 被忽略。要用 **FPU**（父節點 Q 減 0.25），且 virtual loss **不能套在根節點**。

   2 + 3 合計讓 MCTS 少了兩百多 Elo，而修正前 SPRT 照樣判定 H1、勝率 77.6 %。

4. **謎題對照用獨立樣本標準誤** → 同一批題目是配對資料，要用 **McNemar**。
   獨立樣本會把真實差異埋進雜訊。

5. **Elo 信賴區間用 Wald** → 得分率逼近 0 或 1 時區間塌成一點，點估計跑到區間外。

6. **val top-1 領先 ≠ 棋力更強** → base preset 三個指標全面領先，800 局對打卻 +12.6 ± 21.3，無結論。

7. **`train.py` 寫死輸出到 `models/`** → 換 preset 重訓會蓋掉舊 checkpoint 並繞過 SPRT 把關。

8. **自我對弈一代沒跑完就什麼都不留** → 連續兩次中斷、零產出。已改成每 25 局落地。

9. **cp950 編碼** → `src/__init__.py` 統一把 stdout/stderr 轉成 UTF-8，新的進入點不要繞過。

10. **MCTS 是 CPU-bound** → 400 次模擬 143 ms 裡 GPU 只佔 25 ms。
    加大 batch 沒用，要快得靠多行程。

11. **checkpoint 記的架構跟權重不一致** → 存候選模型時寫了 `cfg.to_dict()`，
    但 cfg 來自 config.yaml 的預設 preset（base, C128），權重卻是從 best.pt
    繼承的 small(C96)。下一代載入時照 C128 建模型再載 C96 權重 → size mismatch。
    **存檔要沿用「載進來那個 checkpoint 的 config」，不能拿全域設定去猜。**

12. **`.venv\Scripts\python.exe` 是轉發用的 shim** → 用 `Start-Process` 啟動時，
    回傳的 PID 是外殼（1 執行緒、CPU 0），真正工作的是它的**子行程**。
    看 CPU 判斷死活會誤判；直接看 `data/selfplay` 的檔案時間最可靠。

13. **可見的主控台視窗會被順手關掉** → 工作排程器用 Interactive 身分直接跑 `cmd.exe`
    會開一個視窗杵在桌面上。使用者關掉它 = `CTRL_CLOSE_EVENT`，整個自我對弈死掉，
    結束碼 `0xC000013A`（`STATUS_CONTROL_C_EXIT`）。實測掉了兩次。
    `run_selfplay_detached.ps1` 因此多包一層 VBS：`WScript.Shell.Run(cmd, 0, True)`，
    `0` = 隱藏視窗。**看到 `0xC000013A` 就是被外力砍的，不是程式壞了。**

14. **`.ps1` 沒有 BOM 會被 cp950 解讀** → Windows PowerShell 5.1 讀不含 BOM 的 UTF-8
    檔案時當成 ANSI，中文註解被拆碎後吃掉引號，直接變成語法錯誤
    （`The string is missing the terminator`）。**PowerShell 指令稿一律存成含 BOM 的 UTF-8。**
    這跟 §5 的 9 是同一個 cp950 家族的問題。

15. **煙霧測試污染正式輸出** → `--smoke-test` 原本只換 replay buffer 目錄，
    候選模型與統計 csv 照樣寫進正式路徑。跑一次就把 `selfplay_candidate.pt`
    換成只練了 50 步的模型（下一代會拿它當持續訓練的起點），
    並在 `selfplay_log.csv` 多寫一行 4 局的假資料。兩者都不報錯。
    現在三個輸出全部隔離，抽成 `apply_smoke_test_overrides()` 並有測試守著。
    **測試模式要在路徑層面隔離，不能靠「記得別亂跑」。**

---

## 6. 程式碼風格

- **註解用繁體中文**，技術名詞保留英文；每個函式要有 docstring 說明 shape 與意義
- 完整 type hints
- **不要過度抽象**（讀者剛開始學 class）；能用函式解決就不要開 class，開 class 要在檔頭說明理由
- 不用 metaclass、decorator 魔法、動態 import
- 魔術數字要有具名常數
- 錯誤訊息要指出**下一步該跑什麼指令**
- **視角轉換每次出現都要寫註解說明**
- 不要引入 wandb / tensorboard；csv + matplotlib 就好
- 測試要能在 CPU 上跑，不依賴 checkpoint（用隨機權重的小模型）
