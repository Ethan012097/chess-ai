# CLAUDE-PHASE2.md — 西洋棋 AI 專案 第二階段規格書

## 0. 定位

這份文件涵蓋 **Phase 1 完成之後**的所有工作。基礎規格見 `CLAUDE.md`,兩份文件並存,本文件是它的延伸而非取代。

> **本文件取代先前的 `CLAUDE-ADVANCED.md` 與 `CLAUDE-VISUAL.md`。那兩個檔案請直接刪除**,避免 AI 讀到互相矛盾的規格。

### 前置條件

`CLAUDE.md` 的 M0–M10 完成(M9 待 Stockfish 就位,見第 2 節)。具體必須已有:

- `src/encoding.py` 的 4672 維著法編碼與 canonical orientation,`tests/test_encoding.py` 全過
- 訓練好的監督式模型,val policy top-1 ≥ 45 %
- `src/search/greedy.py` 的 `Searcher` 介面

### 執行順序

順序是依**相依性**排的,不是依難易度。請照這個順序做。

| 節 | 內容 | 估時 | 為什麼排在這 |
|---|---|---|---|
| 1 | PGN 呈現層 | 1 小時 | 定下 `MoveInfo` 結構,第 4 節的網頁箭頭直接沿用 |
| 2 | UCI + Cute Chess | 半天~1 天 | 解掉卡住的 M9;做網頁的期間已經能看它下棋 |
| 3 | 引擎對抗與 Elo 評估 | 半天 | 之後每次改動都要靠它判斷有沒有變強 |
| 4 | 網頁棋盤與策略箭頭 | 1~2 天 | 「看見它在想什麼」的最終形態 |
| 5 | 用 Stockfish 評分重訓 value | 1 天 | MCTS 的地基 |
| 6 | MCTS | 2~3 天 | AlphaZero 的核心 |
| 7 | 自我對弈訓練迴圈 | 3 天+ | 讓它自己變強 |
| 8 | Nibbler 實驗(選配) | 1 小時 | 可能失敗,不算里程碑 |

第 4 節可以與第 5、6 節並行——網頁不影響棋力,棋力也不影響網頁。

---

## 1. PGN 呈現層

所有對局(評估用、人機對弈、自我對弈)一律存成標準 PGN。這是後續所有分析與展示的基礎。

### 1.1 `MoveInfo` 結構定案

這個結構會被第 1、4、6 節共用,一次定好不要再改:

```python
@dataclass
class MoveInfo:
    move: chess.Move              # 實際走的著法
    san: str                      # SAN 表示,給前端顯示用
    value: float                  # value head 輸出,當前走棋方視角,-1~1
    policy_top: list[tuple[str, float]]   # [(uci, prob), ...] 前 8 名,已套 legal mask
    visits: dict[str, int] | None # MCTS 訪問次數;Phase 1 為 None
    elapsed_ms: int
    fen_before: str               # 走這步之前的盤面
```

`policy_top` 取 8 名而非 5 名:網頁預設顯示 5 條箭頭,但要讓使用者能調整,先多存幾個比較省事。

### 1.2 `src/pgn_writer.py`

```python
def write_game(
    board: chess.Board,
    move_infos: list[MoveInfo],
    headers: dict[str, str],
    path: Path,
) -> None: ...
```

每一步寫成兩段註解:

```
1. e4 {[%eval 0.08]} {policy: e4 .41 d4 .22 Nf3 .11 c4 .08 e3 .04 | 0.03s}
```

- 第一段 `[%eval x.xx]` 是 **lichess 認得的標準格式**,單位是兵值,**從白方視角**。把 PGN 貼到 <https://lichess.org/paste> 就會自動畫出評分曲線與分析板
- 第二段是自己的除錯資訊,lichess 會忽略

**視角轉換**:value head 輸出是當前走棋方視角。寫進 `[%eval]` 時要(a)黑方走棋時取負號轉成白方視角;(b)從 tanh 值換算成兵值,見 1.3。

必填 headers:`Event`、`Site`(填 `local`)、`Date`、`White`、`Black`、`Result`、`WhiteElo`/`BlackElo`(對手是 Stockfish 就填其設定),以及自訂的 **`ModelCheckpoint`(填 checkpoint 檔名)**。最後這個很重要,不然三個月後你會分不清這局是哪一版下的。

### 1.3 tanh 值換算成 centipawn

顯示用,不影響棋力。預設用後者:

```python
cp = int(600 * value)                          # 簡單線性
cp = int(290.68 * math.tan(1.548 * value))     # Leela 公式,數值感較貼近 Stockfish
```

`value` 接近 ±1 時 tan 會爆掉,先 clamp 到 ±0.9999,結果再限制在 ±10000 內。

### 1.4 CLI 文字棋盤強化

`play.py --mode cli` 每一步之後印出:

```
  a b c d e f g h
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜  8
...
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖  1
  a b c d e f g h

AI: e4   (信心 41%, 評估 +0.08, 0.03s)
候選: e4 41% | d4 22% | Nf3 11% | c4 8% | e3 4%

你的著法 >
```

輸入同時接受 UCI(`e2e4`)與 SAN(`e4`、`Nf3`):先試 `board.parse_san()`,失敗再試 `parse_uci`。額外支援 `undo`(悔棋兩步)、`fen`、`save`、`quit`。輸入非法著法時印出所有合法著法的 SAN 列表,不要只說「非法」。

### 1.5 `scripts/make_demo_games.py`

讓模型自我對弈 N 局(預設 10),存成 `logs/demo/game_{n}.pgn`。

- 用 `temperature=0.3` 而非 0,否則 10 局會下出一模一樣的棋
- 開局從 `data/openings.pgn` 隨機取(若第 3 節還沒做,就隨機走前 4 個半步)

### 1.6 驗收

把任一 PGN 貼到 <https://lichess.org/paste>,確認:能逐步播放、評分曲線有畫出來、曲線正負方向合理(白方優勢時在上方)。

**最後這項是 value 視角轉換的實地測試,比單元測試更容易抓到搞反。**

---

## 2. UCI 協定與 Cute Chess

實作完這一節,你的引擎就變成標準引擎,可以掛進任何西洋棋 GUI,也可以用 cutechess-cli 自動跑上百局。**這是整份文件投報率最高的一節。**

### 2.1 先把 Stockfish 放進去

**這是現在就該做的第一件事,而且不需要寫任何程式。**

1. 到 <https://stockfishchess.org/download/> 下載 **Windows AVX2** 版(若 CPU 較舊、跑起來報錯,改用 SSE4.1 版)
2. 解壓縮,裡面是一個 `.exe`(檔名類似 `stockfish-windows-x86-64-avx2.exe`)
3. **更名為 `stockfish.exe`**,放到 `C:\code\chess-ai\bin\stockfish.exe`(`bin` 資料夾自己建)
4. 在 PowerShell 驗證:

```powershell
cd C:\code\chess-ai
echo "uci" | .\bin\stockfish.exe
```

看到一長串 `option name ...` 並以 `uciok` 結尾就成功了。

5. 確認 `config.yaml` 的 `stockfish_path` 指向 `bin/stockfish.exe`
6. 跑 `python -m src.evaluate --mode match`

**做完這步 M9 就通過了**,Phase 1 才算真正完成。

### 2.2 協定本體

UCI 是純文字的 stdin/stdout 協定。GUI 把引擎當子行程啟動,一行進、一行出。

`play.py --mode uci` 必須支援:

| 指令 | 回應 | 說明 |
|---|---|---|
| `uci` | `id name ...`、`id author ...`、各 `option`、最後 `uciok` | 握手 |
| `isready` | `readyok` | 任何時候都可能收到,包括思考中 |
| `ucinewgame` | 無 | 清空內部狀態(如 MCTS 樹快取) |
| `position startpos [moves ...]` | 無 | 建立盤面 |
| `position fen <FEN> [moves ...]` | 無 | 同上,從指定 FEN 開始 |
| `go ...` | `info ...` 數行 + `bestmove <uci>` | 開始思考 |
| `stop` | 立即輸出 `bestmove` | 中斷思考 |
| `quit` | 結束行程 | |

`go` 要能解析 `movetime`、`wtime`/`btime`/`winc`/`binc`、`depth`、`nodes`、`infinite`。Phase 1 的 greedy searcher 幾乎瞬間回,可以全部忽略;接上 MCTS 後才需要真的做時間管理(見 6.5)。

要公開的 option:

```
option name Temperature type spin default 0 min 0 max 100
option name TopK type spin default 5 min 1 max 20
option name Checkpoint type string default models/best.pt
option name MCTS type check default false
option name Simulations type spin default 800 min 1 max 100000
```

收到 `setoption name X value Y` 時更新對應設定。

`position` 的處理:**每次都從頭重建 `chess.Board()` 再逐一 push**,不要嘗試增量更新。成本可忽略,但錯誤率差很多。

### 2.3 四個必踩的坑(請在程式碼註解中標明)

**一、stdout 緩衝。** Python 的 stdout 接到 pipe 時是塊緩衝,`print("uciok")` 的字可能根本沒送出去,GUI 等到超時就判引擎當機。在 `main()` 第一行就寫:

```python
sys.stdout.reconfigure(line_buffering=True)
```

**二、stdout 只能有 UCI 訊息。** 任何除錯輸出、tqdm 進度條、PyTorch 的 UserWarning,只要跑到 stdout 就會讓 GUI 解析失敗。所有非協定輸出一律走 `sys.stderr`,並在 import torch 前後用 `warnings.filterwarnings("ignore")` 壓掉警告。

**三、絕對不能吐出非法著法。** GUI 收到非法著法會直接判負,而且不會告訴你原因。`bestmove` 送出前一定要檢查 `move in board.legal_moves`,並包一層 try/except:任何例外都退回「隨便走一步合法著法」,例外訊息印到 stderr,**絕不崩潰、絕不沉默**。

**四、升變與易位格式。** UCI 用長代數式,升變寫 `e7e8q`,易位一律寫王的起訖格 `e1g1`(不是 `e1h1`)。`python-chess` 的 `Move.uci()` / `board.push_uci()` 已處理好,不要自己手刻字串。

### 2.4 info 輸出

```
info depth 1 seldepth 1 score cp 24 nodes 5 nps 400 time 12 pv b8c6
```

`score cp` 是當前走棋方視角的百分兵值,換算見 1.3。將死時改用 `score mate <N>`(正數代表自己 N 步內將死對方)。接上 MCTS 後,`nodes` 填模擬次數,`pv` 填從根節點沿最高訪問次數走下去的路徑。

### 2.5 `engine.bat`

GUI 要的是一個可執行檔,所以在專案根目錄產生:

```bat
@echo off
cd /d C:\code\chess-ai
.venv\Scripts\python.exe -m src.play --mode uci
```

### 2.6 Cute Chess 設定步驟(寫進 README)

1. 到 <https://cutechess.com/> 下載 Windows 版並安裝
2. Tools → Settings → Engines → Add
   - Name:`MyNet`
   - Command:專案根目錄的 `engine.bat`
   - Working Directory:`C:\code\chess-ai`
   - Protocol:`UCI`
3. 同樣方式再加一個 `Stockfish`,Command 指向 `bin\stockfish.exe`
4. 按 OK 時 Cute Chess 會實際啟動引擎確認 `uciok`——**這一步失敗就代表 UCI 有問題**,先看 stderr

三種看它下棋的方式:

- **人機對弈**:Game → New,Player 1 選 Human、Player 2 選 MyNet
- **引擎對打**(推薦,可以坐著看):Player 1 選 MyNet、Player 2 選 Stockfish 並設 `Skill Level=0`,時控 10 秒 + 0.1 秒增秒
- **批次對局**:Tournament → New,在背景跑幾百局

疑難排解(寫進 README):

- **引擎載入失敗** → 多半是 stdout 緩衝或有東西印到 stdout。在 `engine.bat` 尾端加 `2> engine_err.log` 把 stderr 導到檔案再看
- **下到一半引擎消失** → 通常是吐出非法著法後被判負,檢查 stderr
- **反應很慢** → 確認模型只在啟動時載入一次,不是每次 `go` 都重載

### 2.7 `tests/test_uci.py`

用 `subprocess` 啟動自己的引擎,不需要開 GUI 就能驗證:

- 送 `uci`,5 秒內收到 `uciok`
- 送 `isready`,收到 `readyok`
- 送 `position startpos` + `go movetime 100`,回傳的 `bestmove` 是起始盤面的合法著法
- 送一個將死盤面的 FEN,確認引擎不會崩潰也不會 hang
- 送 `quit`,行程在 2 秒內結束

每個測試都要設 timeout,避免 CI 卡死。

---

## 3. 引擎對抗與 Elo 評估

### 3.1 開局書

自動對戰若每局都從起始盤面開始,同一個確定性引擎會下出一模一樣的棋,統計毫無意義。

`scripts/make_openings.py`:從 elite PGN 抽取前 8 個半步,去重後隨機取 2000 個,輸出 `data/openings.pgn`。每個開局在原始資料中至少要出現 50 次,避免抽到冷僻變化。

### 3.2 修正 Elo 信賴區間的算法

**先修這個,再往下做。** 目前 M7 印出的區間有 bug:0 敗時 Elo 上界發散,標準 Wald 區間會退化,導致點估計落在區間外(198勝2和0負 → 點估計 +919 是**正確的**,壞掉的是區間)。

正確做法是先對得分率算 **Wilson 區間**,再把上下端點各自代進 Elo 公式:

```python
def wilson(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """得分率的 Wilson 信賴區間。n 為總局數,和局算 0.5 局。"""
    center = (score + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(score*(1-score)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, center - half), min(1.0, center + half)

def score_to_elo(score: float, cap: float = 1200.0) -> float:
    """得分率轉 Elo 差。score 為 0 或 1 時發散,故 clamp。"""
    score = min(max(score, 1e-6), 1 - 1e-6)
    return max(-cap, min(cap, -400 * math.log10(1/score - 1)))
```

點估計與區間端點**必須經過同一個 `score_to_elo`**,這樣 `assert lower <= point <= upper` 才是有意義的不變式。不要單獨夾住點估計——那是把正確的數字壓成錯的,而且會讓 assert 永遠不響。

### 3.3 `evaluate.py --mode tournament`

組出 cutechess-cli 指令並呼叫、解析輸出:

```
cutechess-cli ^
  -engine name=MyNet cmd=engine.bat ^
  -engine name=SF0 cmd=bin\stockfish.exe "option.Skill Level=0" ^
  -each proto=uci tc=10+0.1 ^
  -games 2 -rounds 100 -repeat ^
  -openings file=data\openings.pgn format=pgn order=random ^
  -pgnout logs\tournament.pgn ^
  -concurrency 2
```

參數意義(寫進註解):

- `-games 2 -rounds 100 -repeat`:每輪打兩局,**同一開局位置先後手各一次**。`-repeat` 不能省,否則白方優勢會污染 Elo 估計
- `-concurrency`:同時跑幾局。**每個引擎實例都會載入一份模型到 GPU**,設太高會 OOM。預設 2
- 含空格的選項名要整個用引號包起來:`"option.Skill Level=0"`

cutechess-cli 跑完會直接印出勝負統計與 Elo 差加誤差範圍,解析最後幾行即可,不要自己重算。

找不到 `cutechess-cli` 時,錯誤訊息要指向 <https://cutechess.com/> 並說明可退回 `--mode match`。

### 3.4 SPRT:判斷新版有沒有比較強

當你有了新 checkpoint,想知道它是否真的比舊版強:

```
-sprt elo0=0 elo1=20 alpha=0.05 beta=0.05
```

意思是「檢定新版是否比舊版強 20 Elo 以上」。它會邊打邊做序貫檢定,一旦統計上有結論就自動停止——通常幾百局就夠,不必固定打滿。

`evaluate.py --mode sprt --new models/epoch_12.pt --old models/best.pt` 包裝這個流程,通過時自動更新 `models/best.pt`。

**沒有 SPRT 就不要相信自己的直覺。** 100 局的勝率差在 ±5 % 以內幾乎沒有統計意義,但人眼看起來會覺得「新版明顯比較強」。

### 3.5 Lichess 謎題測驗(把 top-1 翻譯成人話)

`evaluate.py --mode puzzles`。資料在 `https://database.lichess.org/lichess_db_puzzle.csv.zst`,每題都附一個 Rating。

1. 抽 2000 題,依 Rating 分桶(<1000、1000–1200、⋯、2000+)
2. **每題的 `FEN` 是對手走之前的盤面**,要先 push `Moves` 的第一步才得到要解的局面
3. 讓模型走一步,比對是不是 `Moves` 的第二步
4. 輸出各分數桶的命中率

產出的數字(例如「1000 分以下答對 78 %,1800 分答對 12 %」)是最能對外說明成果的指標,也是之後驗證 MCTS 有沒有用的對照組——同一批題目接上搜尋再跑一次。

**心理準備**:純 policy 網路在深度戰術上會很難看,四步組合殺基本上是猜。這正是 MCTS 要解決的問題。

---

## 4. 網頁棋盤與策略箭頭

棋盤上疊著箭頭,粗細代表模型對每個候選著法的信心,旁邊一條評估條顯示局勢。

### 4.1 技術選型(刻意保持簡單)

- 後端:FastAPI + uvicorn,**重用既有的 `Searcher` 介面**,不要另外寫一套推論邏輯
- 前端:**單一 HTML 檔**,`chessboard.js`(顯示)與 `chess.js`(合法性)都從 CDN 載入
- **不要有建置流程**。沒有 npm、沒有 webpack、沒有 React

理由:這是展示品,不是產品。任何建置步驟都會變成三個月後想重跑時的障礙。

```
src/web/
├── __init__.py
├── server.py          # FastAPI app
└── static/
    └── index.html     # 全部前端程式碼都在這一個檔
```

### 4.2 後端 API

模型在 **app 啟動時載入一次**存成全域,不要每次請求重載。推論端點用**同步 `def`**(不是 `async def`),讓 FastAPI 自動丟到 threadpool,避免阻塞事件迴圈。

```
POST /api/analyse
  request:  { "fen": "...", "top_k": 5 }
  response: {
    "value": 0.12,          # 當前走棋方視角
    "value_white": 0.12,    # 白方視角,前端評估條直接用這個
    "cp": 71,               # 換算後的百分兵值,白方視角
    "moves": [
      {"uci":"e2e4","san":"e4","prob":0.41,"visits":null},
      ...
    ],
    "is_game_over": false,
    "result": null,
    "elapsed_ms": 31
  }

POST /api/move    { fen, uci }  → { fen, legal, is_game_over, result }
GET  /api/health                → { model_path, device, params }
GET  /                          → 回傳 static/index.html
```

**視角處理全部集中在後端**,前端只認 `value_white`,不做任何視角轉換。這是避免搞反最有效的方法。

`/api/health` 要能看出目前載入的是哪個 checkpoint,方便比較不同版本。

### 4.3 前端規格

**版面**:左側垂直評估條(寬 30px)、中間棋盤(560px)、右側資訊面板。

**箭頭繪製**(疊一層 `<svg>` 在棋盤上,`pointer-events: none`):

- 顯示前 `top_k` 名(預設 5,可用滑桿調 1–8)
- 線寬 `width = 4 + 20 * sqrt(prob)`。**用 sqrt 不用線性**——線性的話 3 % 的著法會細到看不見,但那些正是你想觀察的
- 透明度 `opacity = 0.35 + 0.55 * prob`
- 顏色用**同一色相的深淺變化**(統一藍色系,機率越高越深)。不同顏色會讓人誤以為代表不同類別
- 箭頭中點標百分比,`prob < 0.03` 時不標(避免擠成一團)
- 最佳著法額外加粗一級

**評估條**:白色部分高度 = `(value_white + 1) / 2`,中間畫 50 % 參考線,上方顯示 `cp`。

**三種模式**(上方分頁切換):

1. **對弈** — 你拖曳走子,AI 回應。AI 落子前先顯示箭頭停留 800 ms,讓你看到它在考慮什麼
2. **自動播放** — AI 自我對弈,每步間隔可調(預設 1500 ms),可暫停、可單步。**最適合展示**
3. **分析** — 貼上任意 FEN,顯示箭頭與評估,不走棋

**右側資訊面板**:著法列表(SAN,可點擊跳回該步)、當前 FEN(可複製)、每步耗時、匯出 PGN 按鈕。

### 4.4 為 MCTS 預留

接上 MCTS 後,箭頭資料源會從 policy 機率換成**訪問次數比例**。

- `/api/analyse` 回應已有 `visits` 欄位,MCTS 啟用時填入,否則 `null`
- 前端:`visits` 非 null 就用它算比例,否則用 `prob`
- 介面標明目前顯示哪一種(標題寫「Policy 機率」或「MCTS 訪問次數」)

**這兩者意義不同,不能混為一談。** policy 是「直覺想走哪裡」,訪問次數是「想過之後認為哪裡值得」。並排比較是很好的展示點——可以直接看出搜尋修正了直覺的哪些地方。

### 4.5 啟動與驗收

```
uvicorn src.web.server:app --host 127.0.0.1 --port 8000
```

開瀏覽器到 <http://127.0.0.1:8000>。同源,不需要 CORS 設定。

- [ ] 起始盤面最粗的箭頭指向常見開局著法(e4/d4/Nf3),不是奇怪的邊兵
- [ ] 白方明顯優勢的盤面,評估條白色部分明顯過半;黑方優勢時相反(**視角測試**)
- [ ] 拖曳非法著法時棋子彈回,盤面不會進入錯誤狀態
- [ ] 自動播放能跑完一整局並正確顯示結果
- [ ] 匯出的 PGN 貼到 lichess 能正常載入

---

## 5. 用 Stockfish 評分重訓 value head

### 5.1 先更正一個錯誤的期望值

`CLAUDE.md` §7 寫的「value MAE 應降到 0.55–0.62」**是錯的,請忽略那個數字**。

原因:MAE 對「最終誰贏」這種標籤沒有有意義的下界。假設某盤面真實勝率是 0.6,完美的評估器會輸出 +0.2,但標籤只會是 +1 或 −1,誤差恆為 0.8 或 1.2。**就算接上 Stockfish 本人當 value head,對這個 val 集算出的 MAE 也不會低到 0.5。**

實測 0.73(全預測 0 大約是 0.85–0.9)代表它確實學到東西,但這個指標能給的資訊到此為止。**不要再拿 MAE 判斷 value 好壞。**

### 5.2 改用能判斷的指標

`evaluate.py --mode value-quality`:從 val 集抽 3000 個盤面,用 Stockfish depth 12 各評一次,計算:

- 你的 value 與 Stockfish 評分的 **Spearman 等級相關係數**
- **正負號一致率**(誰佔優的判斷有沒有一致)

這直接量到 MCTS 需要的東西——**盤面排序能力**,而不是絕對數值。相關係數低於 0.6 就不要進第 6 節,先把 value 修好。

### 5.3 為什麼要修

MCTS 對 value 品質極度敏感——value 是每次模擬的葉節點評估,歪了會讓搜尋系統性地往錯誤方向展開,比不搜尋還糟。

### 5.4 資料

`https://database.lichess.org/lichess_db_eval.jsonl.zst`,約 3.9 億個盤面,JSONL 格式。壓縮檔數十 GB,**必須串流處理**。

每行結構:`fen` 只含棋子、走棋方、易位權、吃過路兵目標格(沒有步數計數器);`evals` 是一組評估,各有 `knodes`、`depth`、`pvs`,`pvs` 內是 `cp` 或 `mate`。

處理規則:

- 每個盤面**只取 depth 最高那組評估的第一個 pv**
- `depth < 20` 整筆丟棄
- 有 `mate` 的,target 直接設 `±0.99`(正負依 mate 正負)
- 有 `cp` 的,用 1.3 的反函數換算回 tanh 空間:`value = atan(cp / 290.68) / 1.548`,clamp 到 ±0.99
- `fen` 缺步數計數器,補 `0 1`,用 `chess.Board(fen + " 0 1")` 建立
- **視角**:cp 是當前走棋方視角,與 canonical orientation 一致,不需轉換。但這點要**寫測試驗證,不要靠猜**

`--max-positions` 預設 2000 萬。依 FEN 中的棋子總數分桶均勻取樣,確保開中殘局都涵蓋。

### 5.5 訓練方式

**先只重訓 value head,凍結骨幹與 policy head**:

1. 載入 `models/best.pt`
2. `for p in model.parameters(): p.requires_grad = False`,再解凍 value head
3. lr `1e-3`,跑 3 個 epoch,損失只有 `MSELoss`

然後**解凍全部參數微調 2 個 epoch**,lr 降到 `1e-4`,損失回到 `policy_loss + 0.5 * value_loss`(value 權重調低,避免破壞已學好的 policy)。

驗收:5.2 的 Spearman 相關係數應達 0.75 以上;policy top-1 下降不得超過 2 個百分點(掉更多代表微調 lr 太高)。用 3.4 的 SPRT 確認棋力沒退步再覆蓋 `best.pt`。

---

## 6. MCTS

實作 `src/search/mcts.py` 的 `MCTSSearcher`,繼承 `CLAUDE.md` §8 的 `Searcher` 介面。

### 6.1 節點結構

```python
@dataclass
class Node:
    prior: float                  # P,父節點展開時由 policy 給出
    visit_count: int              # N
    value_sum: float              # W
    children: dict[int, "Node"]   # key 是 4672 維的著法 index
    # Q = value_sum / visit_count,visit_count 為 0 時視為 0
```

**不要在 Node 裡存 `chess.Board`。** 那會吃掉大量記憶體。改成搜尋時沿路徑 push/pop,回溯時還原。

### 6.2 一次模擬的四個步驟

**選擇**:從根節點沿 PUCT 分數最高的子節點往下,直到抵達未展開的節點。

```
PUCT(a) = Q(a) + c_puct * P(a) * sqrt(ΣN) / (1 + N(a))
```

`c_puct` 預設 2.0,寫進 config。

> **注意 Q 的視角**:子節點的 Q 是從**子節點走棋方**的視角,父節點要用 `-Q(child)`。**這是 MCTS 最常見的錯誤來源**,請在程式碼註解中明確標示並寫測試驗證。搞反的話 AI 會積極送子,因為它以為那是好事。

**展開**:對葉節點做一次網路前向,得到 policy prior 與 value。policy 先套 legal mask 再 softmax,只為合法著法建立子節點。

**回溯**:沿路徑往上更新 `visit_count += 1`、`value_sum += v`,**每往上一層 v 取負號**(雙人零和)。

**終局處理**:葉節點是將死/和局時不呼叫網路,直接用 `−1` / `0` 當 value。`board.is_game_over()` 要涵蓋 stalemate、五十步、三次重複、子力不足。

### 6.3 根節點的兩個特殊處理

**Dirichlet noise**(只在自我對弈時加,正式對局關掉):

```
P(a) ← (1 - ε) * P(a) + ε * η(a),  η ~ Dir(α),  α = 0.3, ε = 0.25
```

沒有這個噪音,自我對弈會迅速收斂到同一條路線,資料多樣性歸零。

**溫度取樣**:訪問次數分佈 `π(a) ∝ N(a)^(1/τ)`

- 自我對弈:前 30 個半步 `τ = 1.0`(依機率抽樣),之後 `τ → 0`
- 正式對局:`τ = 0`

### 6.4 GPU 批次化(重要)

單執行緒 MCTS 每次模擬只做一次 batch=1 的前向傳播,GPU 使用率會掉到個位數百分比。

**解法:virtual loss + 批次葉節點評估。**

1. 一次選擇 `batch_size`(預設 32)條路徑到葉節點
2. 每選中一個節點就先加 virtual loss(`visit_count += 1`、`value_sum -= 1`),避免同一批全選到同一條路
3. 32 個葉節點盤面一起編碼、一次前向
4. 展開並回溯,同時扣除 virtual loss

這能讓 800 次模擬從幾十秒降到一兩秒。**不做這步,第 7 節的自我對弈會慢到不可行。**

### 6.5 時間管理

`go wtime ... btime ...` 時:`本手可用時間 = 剩餘時間 / 30 + 增秒 * 0.8`,保留 100 ms 安全餘裕。模擬迴圈中每 64 次檢查一次時間與 `stop` 指令。

### 6.6 驗收

- `tests/test_mcts.py`:給一個「一步將死」盤面,100 次模擬必須找到那步
- 給一個「不吃就輸后」的盤面,確認搜尋後選出正確著法
- 視角測試:同一盤面,root 的 `-Q(best_child)` 應與網路直接輸出的 value 同號
- **SPRT 對打 greedy searcher,800 次模擬的 MCTS 應強 200 Elo 以上**
- 3.5 的謎題測驗接上 MCTS 重跑一次,高分桶的命中率應大幅提升

**最後兩項是判斷 MCTS 有沒有寫對的主要依據。若只贏五六十 Elo,value 品質是第一嫌疑犯,回頭做第 5 節。**

---

## 7. 自我對弈訓練迴圈

### 7.1 先講清楚期望值

AlphaZero 從零開始訓練用了 5000 顆 TPU。**你有一張顯卡。**

所以這裡不是從零開始,而是**從監督式模型出發繼續進化**。合理預期:跑一到兩週,棋力提升幾十到一兩百 Elo,然後進入緩慢爬升。這仍然是很好的成果,但不要期待做出 Stockfish。

若跑了三天完全沒進步,先回頭檢查 6.6 的視角測試,而不是加大模型。

### 7.2 先打開兩個 Phase 1 預留的擴充點

**policy target 支援機率分佈**:`CLAUDE.md` §11 預留的 `soft_targets: bool` 現在要真的實作。

- `false`(監督式):target 是整數 index,用 `CrossEntropyLoss`
- `true`(自我對弈):target 是 4672 維機率向量,損失為
  `-(target * log_softmax(logits)).sum(dim=1).mean()`

兩種模式共用同一個訓練迴圈,只在損失計算處分岔。

**稀疏 policy 儲存格式**:4672 維 float32 是 18 KB/盤面,100 萬盤面就 18 GB,不可行。MCTS 的訪問次數分佈本來就極度集中,取前 K 個即可。

```
SPARSE_POLICY_K = 32

pieces      : int8[64]     # 同 Phase 1
castling    : uint8
ep_square   : int8
halfmove    : uint8
top_indices : uint16[32]   # 訪問次數前 32 名的著法 index
top_probs   : uint16[32]   # 機率 × 65535 後取整
value_target: int16        # 值 × 10000
```

約 200 bytes/盤面。載入時展開成 4672 維,未列入處補 0 並重新正規化。合法著法數通常 30–40,K=32 幾乎不會丟掉有意義的機率質量。前處理時要統計「被截斷掉的機率總和」,平均超過 0.02 就調高 K。

### 7.3 迴圈結構

`src/selfplay.py`,一次 iteration:

1. **產生對局**:用當前 `best.pt` 自我對弈 N 局(預設 500),每步 400 次模擬(比正式對局少,換取資料量)
2. **寫入 replay buffer**:用 7.2 的稀疏格式存成 shard
3. **訓練**:從 buffer 抽樣訓練 M 步(預設 2000),`soft_targets: true`
4. **把關**:SPRT 對打舊版,通過才更新 `best.pt`;沒通過就保留舊版,用新資料繼續訓練

### 7.4 Replay buffer

- 保留最近 **20 個 iteration** 的資料,更舊的刪除(滑動視窗,避免被早期的爛資料拖住)
- 抽樣時對較新的 iteration 給較高權重
- 每盤棋的盤面**全部保留**(不像 Phase 1 那樣抽樣),因為自我對弈資料很貴

### 7.5 加速自我對弈的兩個關鍵

**一、多局並行。** 同時開 16 局棋,每局各自跑 MCTS,把所有待評估的葉節點盤面**跨局合併成一個大 batch** 送進 GPU。這比 6.4 的單局批次化再快一個量級。

**二、認輸機制。** 根節點 value 連續 10 步低於 −0.9 就判負結束,省下大量無意義的殘局步數。但要保留 10 % 的對局**不啟用認輸**,用來統計誤判率(應低於 5 %)。

### 7.6 監控

每個 iteration 記錄到 `logs/selfplay_log.csv`:iteration、產生局數、平均局長、和局比例、SPRT 結果、Elo 估計。

**和局比例是最重要的健康指標。** 超過 70 % 代表模型過度保守(常見於 value 主導、policy 多樣性不足),要調高 Dirichlet ε 或溫度。

---

## 8. Nibbler 實驗(選配,可能失敗)

Nibbler(<https://github.com/rooklift/nibbler>)是為 Lc0 設計的介面,策略箭頭是解析引擎的 `info string` 詳細輸出畫出來的,**並不檢查引擎身分**。

理論上模仿 Lc0 的輸出格式就能借用它的介面:

1. 新增 UCI option `VerboseMoveStats type check default false`
2. 開啟時,每次 `go` 之後對每個候選著法輸出一行 `info string`,格式模仿 Lc0 的 `N:`、`(P: xx.xx%)`、`(Q: x.xxxxx)` 欄位

**這個格式相當挑剔,不保證成功。** 試兩次還畫不出箭頭就放棄,第 4 節的自製介面已完全覆蓋這個需求。當成一小時的實驗,不要當里程碑。

---

## 9. 程式碼風格

沿用 `CLAUDE.md` §12 的全部要求:繁體中文註解、完整 type hints、不過度抽象、每支檔案可獨立執行、錯誤訊息要指出下一步。

本文件額外要求:

- **視角轉換(誰的 value、誰的 cp、誰的 Q)每次出現都要寫註解說明。** 這是整個專案錯誤率最高的地方
- MCTS 的每個函式都要在 docstring 說明「這個函式回傳的 value 是從誰的視角」
- 繁中 Windows 的 cp950 編碼問題已在 `src/__init__.py` 統一處理,新增的進入點(`server.py`、`play.py`)不要繞過它

---

## 10. 交付檢查清單

**第 1–2 節(先做完這段,你就能看它下棋了)**

- [ ] **P1** `pgn_writer.py` 與 `MoveInfo` 完成
- [ ] **P2** `make_demo_games.py` 產出 10 局,貼到 lichess 評分曲線方向正確
- [ ] **P3** CLI 模式支援 SAN 輸入、undo、save
- [ ] **P4** `bin/stockfish.exe` 就位,`evaluate.py --mode match` 跑通,**M9 通過**
- [ ] **P5** `play.py --mode uci` 完成,`tests/test_uci.py` 全過
- [ ] **P6** Cute Chess 能載入 `engine.bat` 並完成一局人機對弈
- [ ] **P7** Cute Chess 能跑 MyNet vs Stockfish(Skill 0)引擎對打

**第 3 節**

- [ ] **P8** Elo 信賴區間改用 Wilson,`assert lower <= point <= upper` 成立
- [ ] **P9** `make_openings.py` 產出 2000 個開局
- [ ] **P10** `--mode tournament` 能呼叫 cutechess-cli 並解析 Elo
- [ ] **P11** `--mode sprt` 能判定兩個 checkpoint 的強弱
- [ ] **P12** `--mode puzzles` 產出各分數桶命中率

**第 4 節(可與 5、6 並行)**

- [ ] **P13** `server.py` 啟動,`/api/health` 回傳正確 checkpoint 資訊
- [ ] **P14** 網頁三種模式都可用,4.5 的五項驗收全過

**第 5–7 節**

- [ ] **P15** `--mode value-quality` 完成,取得目前的 Spearman 基準值
- [ ] **P16** Stockfish 評分資料前處理完成
- [ ] **P17** value 重訓完成,Spearman ≥ 0.75 且 policy top-1 掉幅 < 2 %
- [ ] **P18** `mcts.py` 完成,`tests/test_mcts.py` 全過
- [ ] **P19** MCTS(800 模擬)SPRT 對打 greedy,勝出 200 Elo 以上
- [ ] **P20** `selfplay.py` 跑完一個完整 iteration 且 SPRT 有結論
- [ ] **P21** 連續 5 個 iteration,Elo 呈上升趨勢

每完成一項,印出驗收指令,確認通過再繼續。**P19 沒過就不要進 P20**——用歪掉的 MCTS 跑自我對弈只會產生歪掉的資料。
