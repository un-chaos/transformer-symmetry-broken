"""Pytest bootstrap: make the repository root importable as ``transformer_sym``.

Kept deliberately tiny -- it exists so ``python -m pytest`` works from anywhere
without installing the package.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
