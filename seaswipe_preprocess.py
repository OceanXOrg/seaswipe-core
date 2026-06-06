#!/usr/bin/env python
"""Entry point for preprocessing. See `python seaswipe_preprocess.py --help`."""

import sys
from pathlib import Path

# Make `src` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backend.cli.preprocess import main

if __name__ == "__main__":
    raise SystemExit(main())
