"""pytest 設定：確保測試能 `import src.xxx`。

放在專案根目錄的 conftest.py 會讓 pytest 把根目錄加進 sys.path，
所以 `from src.encoding import ...` 才找得到。這裡再明確插一次，
避免不同 pytest 版本的 import mode 造成差異。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
