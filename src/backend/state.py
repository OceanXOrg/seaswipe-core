"""
Filesystem-backed workspace + analysis state.

Replaces the database / task-queue machinery of the full SeaSwipe platform with
a single local directory. One workspace wraps one preprocessing output dir
(``embeddings.npz`` + ``patches/`` + ``manifest.json``) and manages one analysis
*per species*, persisted under ``analyses/{slug}/``.

The server holds a single global Workspace and a single *active* analysis at a
time. This tool is intentionally single-user, single-session — see the README.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np

from src.backend.pipeline.agile_modeling import (
    AnalysisState,
    load_embeddings,
    load_state,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
EMBEDDINGS_NAME = "embeddings.npz"
PATCHES_DIRNAME = "patches"
COCO_NAME = "detections_coco.json"
ANALYSES_DIRNAME = "analyses"


def slugify(name: str) -> str:
    """Filesystem-safe slug for a species name. e.g. 'Sea Pen' -> 'sea-pen'."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "species"


def _label_counts(labels: dict[str, int]) -> tuple[int, int]:
    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = sum(1 for v in labels.values() if v == 0)
    return n_pos, n_neg


class Workspace:
    """One preprocessing output dir + the analyses built on top of it."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.embeddings_path = self.data_dir / EMBEDDINGS_NAME
        self.patches_dir = self.data_dir / PATCHES_DIRNAME
        self.coco_path = self.patches_dir / COCO_NAME
        self.analyses_root = self.data_dir / ANALYSES_DIRNAME

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"No embeddings at {self.embeddings_path}. Run "
                f"`python seaswipe_preprocess.py` first to produce this folder."
            )

        self.manifest = self._load_manifest()
        frames_dir = self.manifest.get("frames_dir")
        self.frames_dir = Path(frames_dir) if frames_dir else None

        logger.info("Loading embeddings from %s", self.embeddings_path)
        self.embeddings, self.filenames = load_embeddings(self.embeddings_path)
        self.filename_to_idx = {f: i for i, f in enumerate(self.filenames)}

        self.analyses_root.mkdir(parents=True, exist_ok=True)
        self._states: dict[str, AnalysisState] = {}
        self.active_slug: str | None = None

    # ── Manifest ─────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        path = self.data_dir / MANIFEST_NAME
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    # ── Analyses ─────────────────────────────────────────────────────────────

    def _analysis_dir(self, slug: str) -> Path:
        return self.analyses_root / slug

    def _meta_path(self, slug: str) -> Path:
        return self._analysis_dir(slug) / "analysis.json"

    def list_analyses(self) -> list[dict]:
        """Lightweight listing of existing analyses (no full state load)."""
        out = []
        for d in sorted(self.analyses_root.glob("*")):
            if not d.is_dir():
                continue
            slug = d.name
            run_id = slug
            species = slug
            meta = self._meta_path(slug)
            if meta.exists():
                try:
                    species = json.loads(meta.read_text()).get("species", slug)
                except Exception:
                    pass
            prefix = f"{run_id}_"
            labels = {}
            lbl_path = d / f"{prefix}labels.json"
            if lbl_path.exists():
                try:
                    labels = json.loads(lbl_path.read_text())
                except Exception:
                    labels = {}
            n_pos, n_neg = _label_counts(labels)
            has_model = (d / f"{prefix}classifier.npz").exists()
            evaluated = False
            eval_path = d / f"{prefix}evaluation.json"
            if eval_path.exists():
                try:
                    evaluated = bool(
                        (json.loads(eval_path.read_text()).get("results") or {}).get("roc_auc")
                        is not None
                    )
                except Exception:
                    evaluated = False
            out.append({
                "species": species,
                "slug": slug,
                "run_id": run_id,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "has_model": has_model,
                "evaluated": evaluated,
            })
        return out

    def select(self, species: str) -> dict:
        """Set the active analysis to ``species`` (creating it if new)."""
        slug = slugify(species)
        if slug not in self._states:
            self._states[slug] = self._load_or_create(species, slug)
        self.active_slug = slug
        return {"species": species, "slug": slug, "run_id": slug}

    def _load_or_create(self, species: str, slug: str) -> AnalysisState:
        output_dir = self._analysis_dir(slug)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Persist the human-readable species name alongside the slug so the
        # listing can show it verbatim.
        meta_path = self._meta_path(slug)
        if not meta_path.exists():
            meta_path.write_text(json.dumps({"species": species, "run_id": slug}, indent=2))

        state = AnalysisState(
            embeddings=self.embeddings,
            filenames=self.filenames,
            filename_to_idx=self.filename_to_idx,
            output_dir=output_dir,
            run_id=slug,
        )

        saved = load_state(output_dir, slug, self.embeddings)
        if "labels" in saved:
            state.labels = saved["labels"]
        if "weights" in saved:
            state.weights = saved["weights"]
            state.bias = saved["bias"]
            state.predictions = saved["predictions"]
        if "eval_labels" in saved:
            state.eval_labels = saved["eval_labels"]
            state.eval_samples = saved.get("eval_samples")
            state.eval_results = saved.get("eval_results")
        if "eval_buckets" in saved:
            state.eval_buckets = saved["eval_buckets"]
        return state

    def active_state(self) -> AnalysisState:
        if self.active_slug is None or self.active_slug not in self._states:
            raise RuntimeError("No active analysis. Select a species first.")
        return self._states[self.active_slug]

    # ── File serving ─────────────────────────────────────────────────────────

    def patch_path(self, filename: str) -> Path | None:
        base = Path(filename).name
        for fp in self.patches_dir.rglob(base):
            if fp.is_file():
                return fp
        return None

    def frame_path(self, filename: str) -> Path | None:
        if self.frames_dir is None or not self.frames_dir.exists():
            return None
        base = Path(filename).name
        for fp in self.frames_dir.rglob(base):
            if fp.is_file():
                return fp
        return None


# ── Module-level singleton ───────────────────────────────────────────────────
# The server is launched against exactly one workspace; the analysis CLI sets it
# before starting uvicorn.

_workspace: Workspace | None = None


def set_workspace(ws: Workspace) -> None:
    global _workspace
    _workspace = ws


def get_workspace() -> Workspace:
    if _workspace is None:
        raise RuntimeError("Workspace not initialised.")
    return _workspace
