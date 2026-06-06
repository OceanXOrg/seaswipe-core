#!/usr/bin/env python
"""Entry point for the analysis web app. See `python seaswipe_analysis.py --help`."""

import sys
from pathlib import Path

# Make `src` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backend.cli.analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
