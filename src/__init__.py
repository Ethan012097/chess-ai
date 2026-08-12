"""西洋棋 AI（監督式學習版）。

Phase 1：用人類高分棋局訓練 policy + value 雙頭網路。
Phase 2：同一個網路接上 MCTS + 自我對弈。
"""

import sys

__version__ = "0.1.0"


def _force_utf8_output() -> None:
    """把 stdout / stderr 切成 UTF-8。

    為什麼需要：繁體中文 Windows 的預設主控台編碼是 cp950，裝不下我們用到的
    幾種符號 —— `✓` `✗`（評估結果）、`♙` `♟`（CLI 棋盤）、`┌` `─`（棋盤外框）、
    `≥`（Elo 上下限）。直接在終端機跑通常沒事（Windows Terminal 會用 UTF-8），
    但只要把輸出**導向檔案或接管線**，Python 就會改用 locale 編碼，
    整個程式會在 print 的時候炸掉 UnicodeEncodeError —— 而且是在跑完 200 局
    評估之後才炸，結果全部白算。

    放在 `src/__init__.py` 是因為不論 `python -m src.xxx` 或
    `python scripts/xxx.py`（會 import src.config）都一定會經過這裡，
    只要設定一次就好。
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        # reconfigure 是 Python 3.7+ 的 TextIOWrapper 方法；被重導向成別的物件
        # （例如測試用的 StringIO）時就跳過，不要硬做。
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # 串流已關閉或不支援重設編碼，維持原狀即可，不值得讓程式起不來
                pass


_force_utf8_output()
