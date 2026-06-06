"""Analysis CLI: launch the SeaSwipe analysis web app for one output directory.

Points at a directory produced by ``seaswipe_preprocess.py`` (containing
``embeddings.npz`` + ``patches/``), then serves the single-page workspace. The
page asks for a species name and runs the full agile-modelling flow; state is
saved under ``<data>/analyses/<species>/`` so relaunching resumes.
"""

import argparse
import logging
from pathlib import Path

import uvicorn

from src.backend.state import Workspace, set_workspace

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("seaswipe.analysis")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seaswipe_analysis.py",
        description="Launch the SeaSwipe analysis workspace for a preprocessed dataset.",
    )
    p.add_argument("--data", type=Path,
                   help="Preprocessing output directory (contains embeddings.npz).")
    p.add_argument("--embeddings", type=Path,
                   help="Path to embeddings.npz directly (its parent dir is used).")
    p.add_argument("--species", default=None,
                   help="Optionally pre-select a species (otherwise the page asks).")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    return p


def run(args: argparse.Namespace) -> int:
    if args.data:
        data_dir = args.data.resolve()
    elif args.embeddings:
        data_dir = args.embeddings.resolve().parent
    else:
        logger.error("Provide --data <dir> (or --embeddings <embeddings.npz>).")
        return 1

    try:
        ws = Workspace(data_dir)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    set_workspace(ws)

    if args.species:
        ws.select(args.species)
        logger.info("Pre-selected species: %s", args.species)

    # Import the app only after the workspace is set; the app object is shared
    # in-process, so we must not use reload / multiple workers.
    from src.backend.main import app, FRONTEND_DIST

    if not FRONTEND_DIST.exists():
        logger.warning(
            "Frontend not built (%s missing). The API will work but the UI "
            "won't load. Run `cd frontend && npm install && npm run build`.",
            FRONTEND_DIST,
        )

    print()
    print(f"SeaSwipe analysis workspace: http://{args.host}:{args.port}")
    print(f"  dataset:  {data_dir}  ({len(ws.filenames)} patches)")
    print(f"  analyses: {ws.analyses_root}")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
