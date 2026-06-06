"""
Agile Modeling — pure functions for interactive image classification.

All functions are stateless: data in, results out. No HTTP, no global state.

The loop: pick a seed patch, label its nearest neighbours, train a linear probe,
inspect the prediction histogram, sample more patches near the decision boundary
(Entropy Sampling), and repeat until the histogram separates the classes.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.backend.settings import AGILE_MODELING_PARAMS, HISTOGRAM_BINS, RANDOM_SEED

logger = logging.getLogger(__name__)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` (write tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path``."""
    _atomic_write_bytes(path, text.encode("utf-8"))


@dataclass
class ClassifierResult:
    weights: np.ndarray
    bias: float
    predictions: np.ndarray
    n_pos: int
    n_neg: int


@dataclass
class AnalysisState:
    """In-memory state for one analysis (one taxon)."""
    embeddings: np.ndarray
    filenames: list[str]
    filename_to_idx: dict[str, int]
    output_dir: Path
    run_id: str
    labels: dict[str, int] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    weights: np.ndarray | None = None
    bias: float = 0.0
    # Evaluation state — a single annotation pass (B buckets x K samples) drives
    # ROC-AUC, the PRF curve, and the Count Error Estimation.
    eval_buckets: list[dict] | None = None
    eval_labels: dict[str, int] = field(default_factory=dict)
    eval_samples: list[list[str]] | None = None
    eval_results: dict | None = None


# ── Core Functions ───────────────────────────────────────────────────────────


def load_embeddings(embeddings_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and filenames from a .npz file."""
    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    filenames = [str(f) for f in data["filenames"]]
    logger.info("Loaded %d embeddings (dim=%d)", len(filenames), embeddings.shape[1])
    return embeddings, filenames


def cosine_search(
    embeddings: np.ndarray,
    query_idx: int,
    k: int,
    n_dissimilar: int = 0,
) -> list[dict]:
    """Return k most similar + n_dissimilar least similar neighbours.

    Assumes embeddings are L2-normalised (so dot product = cosine similarity).
    """
    sims = embeddings @ embeddings[query_idx]
    order = np.argsort(sims)[::-1]
    order = order[order != query_idx]  # exclude self from both ends

    top_k = order[:k]
    results = [
        {"index": int(i), "similarity": round(float(sims[i]), 4)}
        for i in top_k
    ]

    if n_dissimilar > 0:
        bottom = order[-n_dissimilar:][::-1]
        results += [
            {"index": int(i), "similarity": round(float(sims[i]), 4)}
            for i in bottom
        ]

    return results


def train_classifier(
    embeddings: np.ndarray,
    labels: dict[str, int],
    filename_to_idx: dict[str, int],
    params: dict | None = None,
    seed: int = RANDOM_SEED,
) -> ClassifierResult:
    """Train a binary linear probe on the current annotations.

    - Linear head, no hidden layer: ``nn.Linear(D, 1)``
    - Adam lr=0.001, batch_size=32, 128 epochs, BCEWithLogitsLoss
    - Trains on all labels; validation is done separately via the ROC-AUC
      bucket estimator in evaluation.py.
    - Deterministic given the same labels and seed.
    """
    p = {**AGILE_MODELING_PARAMS, **(params or {})}

    torch.manual_seed(seed)

    pos_fns = [f for f, l in labels.items() if l == 1 and f in filename_to_idx]
    neg_fns = [f for f, l in labels.items() if l == 0 and f in filename_to_idx]

    if not pos_fns or not neg_fns:
        raise ValueError("Need at least 1 positive and 1 negative label.")

    indices, targets = [], []
    for f in sorted(labels.keys()):  # sort for determinism across dict orderings
        l = labels[f]
        if f not in filename_to_idx:
            continue
        indices.append(filename_to_idx[f])
        targets.append(l)

    X_train = torch.tensor(embeddings[indices], dtype=torch.float32)
    y_train = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    D = X_train.shape[1]
    linear = nn.Linear(D, 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=p["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    linear.train()
    for _ in range(p["train_epochs"]):
        perm_train = torch.randperm(len(X_train))
        for i in range(0, len(X_train), p["batch_size"]):
            batch = perm_train[i : i + p["batch_size"]]
            loss = criterion(linear(X_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    weights = linear.weight.data.numpy().flatten()
    bias = linear.bias.data.item()
    predictions = run_inference(embeddings, weights, bias)

    return ClassifierResult(
        weights=weights,
        bias=bias,
        predictions=predictions,
        n_pos=len(pos_fns),
        n_neg=len(neg_fns),
    )


def run_inference(embeddings: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Run linear classifier inference on all embeddings. Returns logits."""
    return embeddings @ weights + bias


def margin_sample(
    predictions: np.ndarray,
    filenames: list[str],
    target_logit: float,
    n: int,
    labeled_set: set[str],
) -> list[dict]:
    """Sample patches closest to ``target_logit``, excluding already-labelled ones.

    With ``target_logit = 0`` this is Entropy Sampling for a binary classifier:
    the Bernoulli entropy of sigmoid(logit) is maximal at the decision boundary.
    """
    distances = np.abs(predictions - target_logit)
    order = np.argsort(distances)
    results = []
    for i in order:
        if filenames[i] not in labeled_set:
            results.append({
                "filename": filenames[i],
                "logit": round(float(predictions[i]), 4),
            })
            if len(results) >= n:
                break
    return results


def make_histogram_data(predictions: np.ndarray, n_bins: int = HISTOGRAM_BINS) -> dict:
    """Return histogram bin data as a JSON-serialisable dict (client renders it)."""
    counts, bin_edges = np.histogram(predictions, bins=n_bins)
    bins = []
    for i in range(len(counts)):
        bins.append({
            "bin_start": round(float(bin_edges[i]), 4),
            "bin_end": round(float(bin_edges[i + 1]), 4),
            "count": int(counts[i]),
        })
    return {
        "bins": bins,
        "min": round(float(predictions.min()), 4),
        "max": round(float(predictions.max()), 4),
        "mean": round(float(predictions.mean()), 4),
        "n_positive": int(np.sum(predictions >= 0)),
        "n_negative": int(np.sum(predictions < 0)),
    }


# ── Persistence ──────────────────────────────────────────────────────────────


def _save_classifier_npz(path: Path, weights: np.ndarray, bias: float) -> None:
    """Atomically write a classifier .npz."""
    import io
    buf = io.BytesIO()
    np.savez(buf, weights=weights, bias=np.array([bias]))
    _atomic_write_bytes(path, buf.getvalue())


def _format_predictions_csv(
    filenames: list[str],
    predictions: np.ndarray,
) -> str:
    """Render the predictions CSV body (filename, logit, probability)."""
    logits = np.asarray(predictions, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))  # vectorised sigmoid
    lines = ["filename,logit,probability"]
    for fn, logit, prob in zip(filenames, logits, probs):
        lines.append(f"{fn},{float(logit):.6f},{float(prob):.6f}")
    lines.append("")
    return "\n".join(lines)


def save_state(
    output_dir: Path,
    run_id: str,
    labels: dict[str, int],
    weights: np.ndarray | None,
    bias: float,
    predictions: np.ndarray | None,
    filenames: list[str],
    write_predictions: bool = True,
) -> None:
    """Save analysis state (labels, classifier, predictions) to disk.

    All writes are atomic (temp file + rename). ``write_predictions=False`` skips
    the (potentially large) predictions.csv rewrite — used from the annotate
    path, where only labels change.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_id}_" if run_id else ""

    _atomic_write_text(
        output_dir / f"{prefix}labels.json",
        json.dumps(labels, indent=2),
    )

    if weights is not None:
        _save_classifier_npz(output_dir / f"{prefix}classifier.npz", weights, bias)

    if write_predictions and predictions is not None:
        _atomic_write_text(
            output_dir / f"{prefix}predictions.csv",
            _format_predictions_csv(filenames, predictions),
        )


def save_eval_state(
    output_dir: Path,
    run_id: str,
    eval_labels: dict[str, int],
    eval_samples: list[list[str]] | None,
    eval_buckets: list[dict] | None,
    eval_results: dict | None,
) -> None:
    """Save evaluation state to disk.

    Persists the *full* bucket structure including 'indices' so the
    bucket-to-sample mapping is exactly what eval_start produced; rebuilding from
    current predictions would silently corrupt the ROC-AUC estimate if
    predictions changed between eval start and eval compute.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_id}_" if run_id else ""
    eval_data = {
        "labels": eval_labels,
        "samples": eval_samples,
        "buckets": eval_buckets,  # full structure including indices
        "results": eval_results,
    }
    _atomic_write_text(
        output_dir / f"{prefix}evaluation.json",
        json.dumps(eval_data, indent=2),
    )


def load_state(
    output_dir: Path,
    run_id: str,
    embeddings: np.ndarray,
) -> dict:
    """Load saved analysis state from disk. Returns dict of loaded fields.

    Raises ValueError if a saved classifier's dimension does not match the
    current embeddings — that means the embedding backbone changed under the
    analysis, and the right action is to retrain from scratch.
    """
    prefix = f"{run_id}_" if run_id else ""
    result: dict = {}

    lbl_path = output_dir / f"{prefix}labels.json"
    if lbl_path.exists():
        with open(lbl_path) as f:
            result["labels"] = json.load(f)
        logger.info("Loaded %d labels", len(result["labels"]))

    clf_path = output_dir / f"{prefix}classifier.npz"
    if clf_path.exists():
        data = np.load(clf_path)
        weights = data["weights"]
        if weights.shape[0] != embeddings.shape[1]:
            raise ValueError(
                f"Classifier dim ({weights.shape[0]}) does not match current "
                f"embeddings dim ({embeddings.shape[1]}). The embedding backbone "
                f"likely changed since this classifier was trained. Retrain from "
                f"scratch on the new backbone."
            )
        result["weights"] = weights
        result["bias"] = float(data["bias"][0])
        result["predictions"] = run_inference(embeddings, weights, result["bias"])
        logger.info("Loaded classifier (dim=%d, predictions computed)", weights.shape[0])

    eval_path = output_dir / f"{prefix}evaluation.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        result["eval_labels"] = eval_data.get("labels", {})
        result["eval_samples"] = eval_data.get("samples")
        result["eval_results"] = eval_data.get("results")
        saved_buckets = eval_data.get("buckets")
        if saved_buckets:
            result["eval_buckets"] = saved_buckets
        logger.info(
            "Loaded evaluation state (%d eval labels)",
            len(result.get("eval_labels", {})),
        )

    return result
