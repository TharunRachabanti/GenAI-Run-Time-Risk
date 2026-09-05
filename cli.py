import sys
from pathlib import Path

# Add the backend directory to Python's module search path
# This allows imports like `from app.xxx import yyy` to resolve correctly
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.cli import cli

if __name__ == "__main__":
    cli()
