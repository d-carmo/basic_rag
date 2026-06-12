import sys
from pathlib import Path

# Allow imports from src/ without a full package install
sys.path.insert(0, str(Path(__file__).parent / "src"))
