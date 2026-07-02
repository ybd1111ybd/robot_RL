from pathlib import Path
import runpy
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "runtime" / "lula_action_graph_min.py"

if __name__ == "__main__":
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    runpy.run_path(str(TARGET), run_name="__main__")
