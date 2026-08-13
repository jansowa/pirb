import sys
from pathlib import Path

PIRB_ROOT = Path(__file__).resolve().parents[1]
if str(PIRB_ROOT) not in sys.path:
    sys.path.insert(0, str(PIRB_ROOT))
