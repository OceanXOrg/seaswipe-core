"""Analysis API — the single-analysis agile-modelling workflow.

Operates on the server's one active analysis (selected by taxon). No database,
no auth, no task queue: state lives on the filesystem under the workspace's
``analyses/{slug}/`` directory.
"""

import io
import json
import logging
import random
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from src.backend.models import (
    SelectRequest,
    SearchRequest,
    AnnotateRequest,
    SampleRequest,
    EvalStartRequest,
)
from src.backend.pipeline.agile_modeling import (
    cosine_search,
    train_classifier,
    margin_sample,
    make_histogram_data,
    save_state,
    save_eval_state,
    _format_predictions_csv,
)
from src.backend.pipeline.evaluation import (
    create_log_buckets,
    sample_bucket,
    compute_roc_auc_from_labels,
    compute_prf_curve,
    best_f1_threshold,
    compute_count_error_estimation,
)
from src.backend.settings import RANDOM_SEED, DEFAULT_SAMPLE_SIZE
from src.backend.state import get_workspace

logger = logging.getLogger(__name__)
router = APIRouter()


def _state():
    """Return the active AnalysisState or 400 if none selected yet."""
    try:
        return get_workspace().active_state()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


def _label_counts(labels: dict[str, int]) -> tuple[int, int]:
    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = sum(1 for v in labels.values() if v == 0)
    return n_pos, n_neg


# ── Selection ────────────────────────────────────────────────────────────────


@router.get("/analyses")
def list_analyses():
    """List existing analyses in this workspace (so the user can resume)."""
    return {"analyses": get_workspace().list_analyses()}


@router.post("/select")
def select(body: SelectRequest):
    """Set the active analysis to a taxon (creating it if it doesn't exist)."""
    taxon = body.taxon.strip()
    if not taxon:
        raise HTTPException(400, "Taxon name is required")
    return get_workspace().select(taxon)


@router.get("/current")
def current():
    ws = get_workspace()
    if ws.active_slug is None:
        return {"active": False}
    state = ws.active_state()
    meta = ws._meta_path(ws.active_slug)
    taxon = ws.active_slug
    if meta.exists():
        try:
            taxon = json.loads(meta.read_text()).get("taxon", ws.active_slug)
        except Exception:
            pass
    return {
        "active": True,
        "taxon": taxon,
        "slug": ws.active_slug,
        "run_id": state.run_id,
        "has_model": state.weights is not None,
    }


@router.get("/defaults")
def defaults():
    """Init-phase defaults the frontend uses to seed the controls."""
    from src.backend.settings import AGILE_MODELING_PARAMS

    return {
        "initialisation_nearest_neighbours": AGILE_MODELING_PARAMS["initialisation_nearest_neighbours"],
        "initialisation_n_dissimilar": AGILE_MODELING_PARAMS["initialisation_n_dissimilar"],
        "default_sample_size": DEFAULT_SAMPLE_SIZE,
    }


# ── Stats ────────────────────────────────────────────────────────────────────


@router.get("/stats")
def stats():
    state = _state()
    n_pos, n_neg = _label_counts(state.labels)
    return {
        "n_images": len(state.filenames),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_labeled": len(state.labels),
        "has_model": state.weights is not None,
    }


# ── Search & Annotate ────────────────────────────────────────────────────────


@router.get("/random-patches")
def random_patches(n: int = DEFAULT_SAMPLE_SIZE):
    state = _state()
    k = min(n, len(state.filenames))
    return {"filenames": random.sample(state.filenames, k)}


@router.post("/search")
def search(body: SearchRequest):
    state = _state()
    query = body.query
    if query not in state.filename_to_idx:
        matches = [f for f in state.filenames if query.lower() in f.lower()]
        if not matches:
            raise HTTPException(404, f"No image matching '{query}'")
        query = matches[0]

    neighbors = cosine_search(
        state.embeddings,
        state.filename_to_idx[query],
        body.k,
        body.n_dissimilar,
    )
    for n in neighbors:
        n["filename"] = state.filenames[n["index"]]
    return {"query": query, "neighbors": neighbors}


@router.get("/labels")
def list_labels():
    state = _state()
    return {"labels": dict(state.labels)}


@router.post("/annotate")
def annotate(body: AnnotateRequest):
    state = _state()
    for fn, label in body.labels.items():
        if label is None:
            state.labels.pop(fn, None)
        else:
            state.labels[fn] = int(label)

    # Only labels change here — skip the (potentially large) predictions.csv
    # rewrite. Train rewrites it.
    save_state(
        state.output_dir, state.run_id, state.labels,
        state.weights, state.bias, state.predictions, state.filenames,
        write_predictions=False,
    )
    n_pos, n_neg = _label_counts(state.labels)
    return {"n_pos": n_pos, "n_neg": n_neg, "n_total": len(state.labels)}


# ── Train & Sample ───────────────────────────────────────────────────────────


def _invalidate_eval_state(state) -> None:
    """Wipe eval state in memory and on disk — predictions changed."""
    state.eval_buckets = None
    state.eval_samples = None
    state.eval_labels = {}
    state.eval_results = None
    eval_path = state.output_dir / f"{state.run_id}_evaluation.json"
    if eval_path.exists():
        eval_path.unlink()


@router.post("/train")
def train():
    state = _state()
    try:
        result = train_classifier(
            state.embeddings, state.labels, state.filename_to_idx, seed=RANDOM_SEED,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    state.weights = result.weights
    state.bias = result.bias
    state.predictions = result.predictions

    # Snapshot the labels used for this round so a learning curve can be
    # reconstructed offline (replay each round's labels against the final
    # eval set).
    rounds_dir = state.output_dir / "training_rounds"
    rounds_dir.mkdir(exist_ok=True)
    round_idx = len(list(rounds_dir.glob("round_*.json")))
    with open(rounds_dir / f"round_{round_idx:03d}.json", "w") as f:
        json.dump({
            "round": round_idx,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": state.labels,
            "n_pos": result.n_pos,
            "n_neg": result.n_neg,
        }, f, indent=2)

    # Any prior eval state was drawn against the old classifier's predictions
    # and is now stale.
    _invalidate_eval_state(state)

    save_state(
        state.output_dir, state.run_id, state.labels,
        state.weights, state.bias, state.predictions, state.filenames,
        write_predictions=True,
    )
    return {"n_pos": result.n_pos, "n_neg": result.n_neg}


@router.get("/histogram")
def histogram():
    state = _state()
    if state.predictions is None:
        raise HTTPException(404, "No predictions yet — train the classifier first")
    return make_histogram_data(state.predictions)


@router.post("/sample")
def sample(body: SampleRequest):
    state = _state()
    if state.predictions is None:
        raise HTTPException(400, "Train the classifier first")
    results = margin_sample(
        state.predictions, state.filenames,
        body.target_logit, body.n,
        set(state.labels.keys()),
    )
    return {"images": results, "target_logit": body.target_logit}


@router.get("/positives")
def list_positives(
    threshold: float = 1.0,
    page: int = 0,
    page_size: int = 24,
    sort: str = "desc",
):
    state = _state()
    if state.predictions is None:
        raise HTTPException(404, "No predictions yet — train the classifier first")

    above_idx = np.where(state.predictions >= threshold)[0]
    scores = state.predictions[above_idx]
    order = np.argsort(scores if sort == "asc" else -scores)
    sorted_idx = above_idx[order]

    total = int(len(sorted_idx))
    start = page * page_size
    page_idx = sorted_idx[start : start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "threshold": threshold,
        "patches": [
            {
                "filename": state.filenames[int(i)],
                "logit": round(float(state.predictions[int(i)]), 4),
            }
            for i in page_idx
        ],
    }


# ── Evaluation ───────────────────────────────────────────────────────────────


@router.post("/eval/start")
def eval_start(body: EvalStartRequest):
    state = _state()
    if state.predictions is None:
        raise HTTPException(400, "Train the classifier first")

    state.eval_buckets = create_log_buckets(
        state.predictions, state.filenames, body.n_buckets,
        exclude_filenames=set(state.labels.keys()),
    )
    state.eval_labels = {}
    state.eval_samples = [
        sample_bucket(b, body.k, state.filenames, seed=RANDOM_SEED)
        for b in state.eval_buckets
    ]
    state.eval_results = None

    save_eval_state(
        state.output_dir, state.run_id,
        state.eval_labels, state.eval_samples, state.eval_buckets, state.eval_results,
    )
    return {
        "n_buckets": len(state.eval_buckets),
        "k": body.k,
        "total_to_label": sum(len(s) for s in state.eval_samples),
        "buckets": [{
            "bucket": b["bucket"],
            "size": b["size"],
            "score_min": round(b["score_min"], 4),
            "score_max": round(b["score_max"], 4),
            "n_sampled": len(state.eval_samples[i]),
        } for i, b in enumerate(state.eval_buckets)],
    }


@router.get("/eval/bucket/{bucket_idx}")
def eval_bucket(bucket_idx: int):
    state = _state()
    if state.eval_samples is None:
        raise HTTPException(400, "Start evaluation first")
    if bucket_idx < 0 or bucket_idx >= len(state.eval_samples):
        raise HTTPException(400, f"Invalid bucket index {bucket_idx}")

    samples = state.eval_samples[bucket_idx]
    bucket = state.eval_buckets[bucket_idx]
    return {
        "bucket": bucket_idx,
        "n_buckets": len(state.eval_buckets),
        "size": bucket["size"],
        "score_min": round(bucket["score_min"], 4),
        "score_max": round(bucket["score_max"], 4),
        "images": samples,
        "labels": {fn: state.eval_labels[fn] for fn in samples if fn in state.eval_labels},
    }


@router.post("/eval/annotate")
def eval_annotate(body: AnnotateRequest):
    state = _state()
    if state.eval_samples is None:
        raise HTTPException(400, "Start evaluation first")

    for fn, label in body.labels.items():
        if label is None:
            state.eval_labels.pop(fn, None)
        else:
            state.eval_labels[fn] = int(label)

    save_eval_state(
        state.output_dir, state.run_id,
        state.eval_labels, state.eval_samples, state.eval_buckets, state.eval_results,
    )

    progress, total_labeled, total_samples = [], 0, 0
    for i, samples in enumerate(state.eval_samples):
        labeled = sum(1 for fn in samples if fn in state.eval_labels)
        progress.append({"bucket": i, "labeled": labeled, "total": len(samples)})
        total_labeled += labeled
        total_samples += len(samples)
    return {"progress": progress, "total_labeled": total_labeled, "total_samples": total_samples}


@router.post("/eval/compute")
def eval_compute():
    state = _state()
    if state.eval_buckets is None or state.eval_samples is None:
        raise HTTPException(400, "Start evaluation first")

    for i, samples in enumerate(state.eval_samples):
        labeled = sum(1 for fn in samples if fn in state.eval_labels)
        if labeled == 0:
            raise HTTPException(400, f"Bucket {i} has no labels yet")

    results = compute_roc_auc_from_labels(state.eval_buckets, state.eval_samples, state.eval_labels)
    prf = compute_prf_curve(
        state.eval_buckets, state.eval_samples, state.eval_labels, state.predictions,
    )
    best_t = best_f1_threshold(prf)
    count_error = None
    if best_t is not None:
        count_error = compute_count_error_estimation(
            state.eval_buckets, state.eval_samples, state.eval_labels,
            state.predictions, state.filename_to_idx, best_t,
        )
    results["prf_curve"] = prf
    results["best_threshold"] = best_t
    results["count_error"] = count_error
    state.eval_results = results

    save_eval_state(
        state.output_dir, state.run_id,
        state.eval_labels, state.eval_samples, state.eval_buckets, state.eval_results,
    )
    return results


@router.get("/eval/status")
def eval_status():
    state = _state()
    if state.eval_buckets is None:
        return {"active": False}

    progress, total_labeled, total_samples = [], 0, 0
    for i, samples in enumerate(state.eval_samples):
        labeled = sum(1 for fn in samples if fn in state.eval_labels)
        progress.append({"bucket": i, "labeled": labeled, "total": len(samples)})
        total_labeled += labeled
        total_samples += len(samples)
    return {
        "active": True,
        "n_buckets": len(state.eval_buckets),
        "progress": progress,
        "total_labeled": total_labeled,
        "total_samples": total_samples,
        "results": state.eval_results,
    }


# ── Export ───────────────────────────────────────────────────────────────────


@router.get("/predictions/export")
def export_predictions(threshold: float):
    """Stream a CSV of predictions with logit >= ``threshold``."""
    state = _state()
    if state.predictions is None:
        raise HTTPException(404, "No predictions yet")

    mask = np.asarray(state.predictions) >= threshold
    filenames = [fn for fn, keep in zip(state.filenames, mask) if keep]
    preds = state.predictions[mask]
    body = _format_predictions_csv(filenames, preds)
    return StreamingResponse(
        io.BytesIO(body.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={state.run_id}_predictions.csv"},
    )


@router.get("/export")
def export_analysis():
    """Export the full analysis as a ZIP (labels, classifier, predictions, eval)."""
    state = _state()
    prefix = f"{state.run_id}_"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for suffix in ("labels.json", "classifier.npz", "predictions.csv", "evaluation.json"):
            p = state.output_dir / f"{prefix}{suffix}"
            if p.exists():
                zf.write(p, f"{prefix}{suffix}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={state.run_id}_export.zip"},
    )


# ── Patch / Frame serving ────────────────────────────────────────────────────


def _parse_patch_annotation_id(filename: str) -> int | None:
    """Extract the annotation id from a patch filename '{frame_stem}_patch{N}.jpg'."""
    stem = Path(filename).stem
    idx = stem.rfind("_patch")
    if idx < 0:
        return None
    try:
        return int(stem[idx + len("_patch"):])
    except ValueError:
        return None


def _lookup_patch_source(coco_path: Path, annotation_id: int) -> dict:
    if not coco_path.is_file():
        raise HTTPException(404, f"detections_coco.json not found: {coco_path}")
    with open(coco_path) as f:
        coco = json.load(f)
    annotation = next(
        (a for a in coco.get("annotations", []) if a.get("id") == annotation_id), None
    )
    if annotation is None:
        raise HTTPException(404, f"Annotation {annotation_id} not found")
    image = next(
        (i for i in coco.get("images", []) if i.get("id") == annotation["image_id"]), None
    )
    if image is None:
        raise HTTPException(404, f"Image {annotation['image_id']} not found")
    return {
        "file_name": image["file_name"],
        "width": image["width"],
        "height": image["height"],
        "bbox": annotation["bbox"],
    }


@router.get("/patches/{filename}/source")
def patch_source(filename: str):
    """Return the source frame + bbox for a patch (used by the lightbox)."""
    ws = get_workspace()
    annotation_id = _parse_patch_annotation_id(filename)
    if annotation_id is None:
        raise HTTPException(400, f"Not a valid patch filename: {filename}")
    info = _lookup_patch_source(ws.coco_path, annotation_id)
    frame_available = ws.frame_path(info["file_name"]) is not None
    return {
        "frame_filename": info["file_name"],
        "frame_url": f"/api/frames/{info['file_name']}" if frame_available else None,
        "bbox": info["bbox"],
        "frame_width": info["width"],
        "frame_height": info["height"],
    }


@router.get("/patches/{filename:path}")
def serve_patch(filename: str):
    fp = get_workspace().patch_path(filename)
    if fp is None:
        raise HTTPException(404, "Patch not found")
    return FileResponse(fp, media_type="image/jpeg")


@router.get("/frames/{filename:path}")
def serve_frame(filename: str):
    fp = get_workspace().frame_path(filename)
    if fp is None:
        raise HTTPException(404, "Frame not found")
    return FileResponse(fp, media_type="image/jpeg")
