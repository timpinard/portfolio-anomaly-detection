import sys
from pathlib import Path

# Add <repo_root>/src to sys.path so tests can import modules under src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
