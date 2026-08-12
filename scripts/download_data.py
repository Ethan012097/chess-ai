"""下載訓練用的 PGN 棋譜。

用法（在專案根目錄執行）：
    python scripts/download_data.py --source elite --month 2024-01
    python scripts/download_data.py --source elite --month 2024-01 --sample
    python scripts/download_data.py --source pgnmentor           # 小檔，煙霧測試用
    python scripts/download_data.py --list                       # 只列出可用來源

四種來源：
    elite     Lichess Elite Database（已篩過 2000+ 分，預設，最推薦）
    lichess   Lichess 完整月檔（.pgn.zst，單月可達 30 GB，串流解壓）
    ccrl      引擎對局（PGN 內含每步評分，Phase 2 可當 value 的額外標註）
    pgnmentor 大師棋譜（檔案小、乾淨，適合快速煙霧測試）

`.zst` 一律用 zstandard 的 stream reader 邊解壓邊寫檔，不會整檔解到硬碟再處理。
下載支援續傳：檔案已存在就跳過（用 --force 可強制重抓）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# 讓這支腳本可以直接用 `python scripts/download_data.py` 執行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.config import load_config  # noqa: E402

DOWNLOAD_CHUNK_SIZE = 1 << 20        # 1 MB，下載與解壓的緩衝區大小
SAMPLE_GAME_LIMIT = 5000             # --sample 模式只留前 5000 局
DEFAULT_MONTH = "2025-11"            # elite 資料庫目前最新的月份

# 各壓縮格式的檔頭魔術位元組，用來擋掉「HTTP 200 但其實是 HTML 錯誤頁」的情況
MAGIC_BYTES: dict[str, bytes] = {
    ".zip": b"PK\x03\x04",
    ".zst": b"\x28\xb5\x2f\xfd",
}

# 各來源的說明，`--list` 會印出來
SOURCES: dict[str, str] = {
    "elite": "Lichess Elite Database（2500+ 對 2300+ 已篩選、已排除 bullet，單月約 70–80 MB）",
    "lichess": "Lichess 官方完整月檔（.pgn.zst，單月壓縮就約 28 GB，CC0 授權）",
    "ccrl": "CCRL 引擎對局（含每步評分，Phase 2 用）",
    "pgnmentor": "PGN Mentor 大師棋譜（檔案小、乾淨，煙霧測試用）",
}

# elite 資料庫目前最新的月份。2025-12 之後的網址會回 HTTP 200 但內容是 HTML
# 錯誤頁（見 download_file 的檢查），所以預設值要指向真的存在的月份。
LATEST_ELITE_MONTH = "2025-11"

# 各來源的壓縮格式不同，解壓方式也不同：
#   elite     → .zip（Python 標準庫 zipfile）
#   lichess   → .pgn.zst（zstandard 串流解壓）
#   ccrl      → .7z（標準庫解不開，要手動解）
#   pgnmentor → .zip
# 下載完之後會依副檔名自動分流，不要混用。


def build_url(source: str, month: str) -> str:
    """組出下載網址。

    Args:
        source: elite / lichess / ccrl / pgnmentor。
        month: "YYYY-MM"，只有 elite 與 lichess 會用到。

    Returns:
        可直接 GET 的 URL。
    """
    year, mon = month.split("-")
    if source == "elite":
        # https://database.nikonoel.fr/ 的檔名格式：lichess_elite_2024-01.zip
        return f"https://database.nikonoel.fr/lichess_elite_{year}-{mon}.zip"
    if source == "lichess":
        return (
            "https://database.lichess.org/standard/"
            f"lichess_db_standard_rated_{year}-{mon}.pgn.zst"
        )
    if source == "ccrl":
        # CCRL 4040 的完整棋譜打包檔
        return "https://computerchess.org.uk/ccrl/4040/CCRL-4040.[2200000].pgn.7z"
    if source == "pgnmentor":
        # 世界冠軍棋譜合輯，約數 MB
        return "https://www.pgnmentor.com/players/Carlsen.zip"
    raise ValueError(f"不認識的來源：{source}")


def target_filename(source: str, month: str, url: str) -> str:
    """決定要存成什麼檔名。"""
    suffix = "".join(Path(url.split("/")[-1]).suffixes)
    if source in ("elite", "lichess"):
        return f"{source}_{month}{suffix}"
    return url.split("/")[-1]


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """下載檔案，有進度條與續傳（已存在就跳過）。

    Args:
        url: 下載網址。
        dest: 目標路徑。
        force: True 表示即使檔案已存在也重抓。

    Returns:
        下載完成的檔案路徑。
    """
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1 << 20)
        print(f"[skip] {dest.name} 已存在（{size_mb:.1f} MB），跳過下載。")
        print(f"       想重抓請加 --force")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"[下載] {url}")
    # 有些站台（例如 pgnmentor）會擋掉沒有 User-Agent 的請求
    headers = {"User-Agent": "Mozilla/5.0 (chess-ai dataset downloader)"}
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except requests.RequestException as exc:
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(
            f"\n下載失敗：{exc}\n"
            f"下一步：\n"
            f"  1. 用瀏覽器開 {url} 確認檔案還在（來源網站偶爾會改檔名格式）\n"
            f"  2. 或手動下載後放到 {dest}\n"
            f"  3. 或換一個月份：--month 2023-12"
        ) from exc

    # 檢查檔頭：nikonoel.fr 對不存在的月份會回 HTTP 200 + HTML 錯誤頁，
    # 不檢查的話會存下一個副檔名是 .zip 的網頁，等到解壓才報出莫名其妙的錯。
    expected_magic = MAGIC_BYTES.get(dest.suffix.lower())
    if expected_magic is not None:
        with open(tmp, "rb") as f:
            head = f.read(len(expected_magic))
        if head != expected_magic:
            preview = head.decode("utf-8", errors="replace")
            tmp.unlink()
            raise SystemExit(
                f"\n下載到的不是 {dest.suffix} 檔案（開頭是 {preview!r}）。\n"
                f"這通常代表該月份不存在，伺服器回了一個 HTML 錯誤頁。\n"
                f"下一步：\n"
                f"  1. 換一個月份，elite 目前最新是 {LATEST_ELITE_MONTH}：\n"
                f"     python scripts/download_data.py --source {'elite'} --month {LATEST_ELITE_MONTH}\n"
                f"  2. 或用瀏覽器開 https://database.nikonoel.fr/ 看有哪些月份"
            )

    tmp.replace(dest)
    size_mb = dest.stat().st_size / (1 << 20)
    print(f"[完成] {dest}（{size_mb:.1f} MB）")
    return dest


def decompress_zst(src: Path, dest: Path, max_games: int | None = None) -> Path:
    """串流解壓 .zst，邊解邊寫，不會把整包解到記憶體或硬碟暫存。

    Args:
        src: .pgn.zst 檔案。
        dest: 解出來的 .pgn 路徑。
        max_games: 只留前 N 局（--sample 用），None 表示全部。

    Returns:
        解壓後的 .pgn 路徑。
    """
    import zstandard

    if dest.exists():
        print(f"[skip] {dest.name} 已解壓，跳過。")
        return dest

    print(f"[解壓] {src.name} → {dest.name}（串流）")
    dctx = zstandard.ZstdDecompressor()
    games_seen = 0

    # "[Event " 有可能剛好被切在兩個 chunk 的交界，所以每次保留上一塊的尾巴
    # 一起數，才不會漏算。長度取標籤長度 - 1 就夠。
    tag = b"[Event "
    overlap = len(tag) - 1

    with open(src, "rb") as fin, open(dest, "wb") as fout:
        with dctx.stream_reader(fin) as reader, tqdm(
            unit="B", unit_scale=True, unit_divisor=1024, desc="解壓"
        ) as bar:
            tail = b""
            while True:
                chunk = reader.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bar.update(len(chunk))
                fout.write(chunk)

                if max_games is None:
                    continue

                # --sample：數 [Event 標籤來判斷已經寫了幾局
                games_seen += (tail + chunk).count(tag)
                if games_seen >= max_games:
                    print(f"\n[sample] 已達 {max_games} 局，停止解壓。")
                    break
                tail = chunk[-overlap:]

    print(f"[完成] {dest}（{dest.stat().st_size / (1 << 20):.1f} MB）")
    return dest


def extract_zip(src: Path, dest_dir: Path, max_games: int | None = None) -> list[Path]:
    """解開 .zip，回傳裡面所有 .pgn 的路徑。

    Args:
        src: .zip 檔案。
        dest_dir: 解壓目的資料夾。
        max_games: 只留前 N 局（--sample 用）。

    Returns:
        解出來的 .pgn 檔案清單。
    """
    import zipfile

    print(f"[解壓] {src.name} → {dest_dir}")
    outputs: list[Path] = []
    with zipfile.ZipFile(src) as zf:
        pgn_names = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
        if not pgn_names:
            raise SystemExit(f"{src.name} 裡面沒有 .pgn 檔案，內容為：{zf.namelist()[:10]}")
        for name in pgn_names:
            out = dest_dir / Path(name).name
            if out.exists():
                print(f"[skip] {out.name} 已存在。")
                outputs.append(out)
                continue
            with zf.open(name) as fin, open(out, "wb") as fout:
                if max_games is None:
                    while True:
                        chunk = fin.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        fout.write(chunk)
                else:
                    games_seen = 0
                    while games_seen < max_games:
                        chunk = fin.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        games_seen += chunk.count(b"[Event ")
                        fout.write(chunk)
                    print(f"[sample] 只保留約前 {max_games} 局。")
            print(f"[完成] {out}（{out.stat().st_size / (1 << 20):.1f} MB）")
            outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下載訓練用 PGN 棋譜",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="elite",
        choices=sorted(SOURCES),
        help="資料來源（預設 elite）",
    )
    parser.add_argument(
        "--month", type=str, default=DEFAULT_MONTH, help="月份 YYYY-MM（elite / lichess 用）"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=f"只取前 {SAMPLE_GAME_LIMIT} 局，讓整條 pipeline 5 分鐘內跑通",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="輸出資料夾（預設 data/raw）")
    parser.add_argument("--force", action="store_true", help="即使檔案已存在也重新下載")
    parser.add_argument("--list", action="store_true", help="列出可用來源後結束")
    parser.add_argument("--config", type=str, default=None, help="config.yaml 路徑")
    args = parser.parse_args()

    if args.list:
        print("可用的資料來源：")
        for name, desc in SOURCES.items():
            mark = "（預設）" if name == "elite" else ""
            print(f"  {name:10s} {desc}{mark}")
        print(f"\nelite 目前最新月份：{LATEST_ELITE_MONTH}")
        print("  單月約 11,000,000 個盤面。規格的目標是 1500 萬–3000 萬，")
        print("  所以建議抓 2–3 個月再一起丟給 preprocess.py：")
        print('    python -m src.preprocess --input "data/raw/*.pgn"')
        print("\nPhase 1.5 備案（目前不實作）：")
        print("  https://database.lichess.org/lichess_db_eval.jsonl.zst（約 20 GB）")
        print("  Lichess 的 Stockfish 評分資料庫，約 3.9 億個已評估盤面。")
        print("  用「最終誰贏」當 value target 其實很吵——優勢方可能後來超時輸掉，")
        print("  那個盤面就被標成錯的。改用 Stockfish 評分當 value target 會乾淨很多。")
        print("  等 value MAE 卡在 0.6 附近下不去的時候再考慮換。")
        print("\nStockfish 不會自動下載，請到 https://stockfishchess.org/download/")
        print("抓 Windows 版，解壓後把執行檔放到 bin/stockfish.exe")
        return

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.resolve_path(cfg.data.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_games = SAMPLE_GAME_LIMIT if args.sample else None
    url = build_url(args.source, args.month)
    dest = out_dir / target_filename(args.source, args.month, url)

    if args.source == "ccrl":
        print(
            "注意：CCRL 提供的是 .7z，Python 標準庫解不開。\n"
            "下載後請用 7-Zip 手動解壓到 data/raw/，再跑 preprocess.py。"
        )

    downloaded = download_file(url, dest, force=args.force)

    # 依副檔名決定要不要解壓
    produced: list[Path] = []
    name = downloaded.name.lower()
    if name.endswith(".zst"):
        produced = [decompress_zst(downloaded, downloaded.with_suffix(""), max_games)]
    elif name.endswith(".zip"):
        produced = extract_zip(downloaded, out_dir, max_games)
    elif name.endswith(".pgn"):
        produced = [downloaded]
    else:
        print(f"[提醒] {downloaded.name} 需要手動解壓到 {out_dir}")

    if produced:
        print("\n可用的 PGN 檔案：")
        for p in produced:
            print(f"  {p}")
        print("\n下一步，執行前處理：")
        pattern = str(out_dir / "*.pgn")
        if args.sample:
            print(f'  python -m src.preprocess --input "{pattern}" --max-positions 200000')
        else:
            print(f'  python -m src.preprocess --input "{pattern}"')


if __name__ == "__main__":
    main()
