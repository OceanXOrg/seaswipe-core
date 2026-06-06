"""
Evaluation — quantile-based ROC-AUC estimation, bucket-density PRF curves,
and Count Error Estimation.

Pure computation functions with no HTTP or database dependencies. All
stochastic samplers accept a `seed` parameter so an evaluation run is
fully reproducible from the same predictions and seed. The default comes
from settings.RANDOM_SEED so the whole pipeline shares one knob.

The full evaluation pipeline reuses a single annotation pass: B
logarithmic quantile buckets, K samples each (default B=6, K=40 → 240
labels). From those annotations we estimate:

  - ROC-AUC via pairwise bucket scoring (Navine et al. 2024)
  - precision/recall/F1 curves across thresholds via within-bucket
    apportionment of P(+|b) onto the score distribution
  - true / false positive / false negative counts at the F1-optimal
    threshold (Count Error Estimation)

The within-bucket apportionment for FN follows the method used in the
SeaSwipe paper companion script
`phase2_quality_estimation/run_phase2a.py` (Navine et al. 2024).
"""

import numpy as np

from src.backend.settings import RANDOM_SEED


def create_log_buckets(
    predictions: np.ndarray,
    filenames: list[str],
    n_buckets: int,
    exclude_filenames: set[str] | None = None,
) -> list[dict]:
    """Create logarithmic quantile buckets over scored predictions.

    Buckets are built bottom-up: the lowest-scoring 50% of examples go into
    bucket 0, the next 25% into bucket 1, etc. The last bucket gets whatever
    fraction remains after B-1 halvings.

    `exclude_filenames`, when provided, drops those patches before bucketing
    so the resulting AUC estimate is over the *unlabelled* portion of the
    dataset only. This is how the API surfaces it: training-labelled patches
    have known truth and would only re-ask the user the same question, so we
    estimate the classifier's quality on what's left. The `indices` stored on
    each bucket still reference positions in the *original* `predictions` /
    `filenames` arrays so `sample_bucket` can map them back to filenames.
    """
    n_full = len(predictions)
    if exclude_filenames:
        keep_mask = np.array(
            [fn not in exclude_filenames for fn in filenames], dtype=bool
        )
        original_indices = np.flatnonzero(keep_mask)
    else:
        original_indices = np.arange(n_full)

    preds = predictions[original_indices]
    n_total = len(preds)
    order = np.argsort(preds)  # ascending by score within the kept subset

    buckets = []
    consumed = 0
    for b in range(n_buckets):
        if b < n_buckets - 1:
            frac = 0.5 ** (b + 1)
            size = int(round(frac * n_total))
        else:
            size = n_total - consumed  # remainder

        if size <= 0:
            continue

        local = order[consumed : consumed + size]
        global_idx = original_indices[local]
        scores = preds[local]
        buckets.append({
            "bucket": b,
            "start_idx": int(consumed),
            "size": int(size),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "indices": global_idx.tolist(),
        })
        consumed += size

    return buckets


def sample_bucket(
    bucket: dict,
    k: int,
    filenames: list[str],
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Randomly sample k filenames from a bucket (without replacement).

    Deterministic given (bucket, k, seed). The seed is salted with the bucket
    index so different buckets in the same evaluation run draw independent
    samples even though they share the base seed.
    """
    indices = np.array(bucket["indices"])
    k = min(k, len(indices))
    rng = np.random.default_rng(seed + int(bucket.get("bucket", 0)))
    chosen = rng.choice(indices, size=k, replace=False)
    return [filenames[i] for i in chosen]


def compute_roc_auc(buckets: list[dict], p_pos_per_bucket: np.ndarray) -> float:
    """Compute ROC-AUC from bucket positive-rates using the pairwise formula.

    p_pos_per_bucket[b] = P(+|b), the estimated positive rate in bucket b.
    Buckets are ordered from lowest-scoring (b=0) to highest-scoring.
    """
    B = len(buckets)
    sizes = np.array([b["size"] for b in buckets], dtype=float)

    est_pos = p_pos_per_bucket * sizes
    est_neg = (1.0 - p_pos_per_bucket) * sizes

    total_pos = est_pos.sum()
    total_neg = est_neg.sum()

    if total_pos == 0 or total_neg == 0:
        return float("nan")

    p_b_given_pos = est_pos / total_pos
    p_b_given_neg = est_neg / total_neg

    auc = 0.0
    for b in range(B):
        for c in range(B):
            if b > c:
                prob = 1.0
            elif b < c:
                prob = 0.0
            else:
                prob = 0.5
            auc += prob * p_b_given_pos[b] * p_b_given_neg[c]

    return float(auc)


def _p_pos_per_bucket(
    buckets: list[dict],
    eval_samples: list[list[str]],
    eval_labels: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (P(+|b), n_pos[b], n_neg[b]) from per-bucket annotations.

    Skipped (unlabelled) samples are not counted. Empty buckets get P=0.
    """
    B = len(buckets)
    n_pos = np.zeros(B)
    n_neg = np.zeros(B)
    for b_idx, samples in enumerate(eval_samples):
        for fn in samples:
            if fn in eval_labels:
                if eval_labels[fn] == 1:
                    n_pos[b_idx] += 1
                else:
                    n_neg[b_idx] += 1
    k_per_bucket = n_pos + n_neg
    p_pos = np.where(k_per_bucket > 0, n_pos / k_per_bucket, 0.0)
    return p_pos, n_pos, n_neg


def compute_roc_auc_from_labels(
    buckets: list[dict],
    eval_samples: list[list[str]],
    eval_labels: dict[str, int],
) -> dict:
    """Compute ROC-AUC point estimate from per-bucket evaluation labels."""
    p_pos, n_pos, n_neg = _p_pos_per_bucket(buckets, eval_samples, eval_labels)
    auc_point = compute_roc_auc(buckets, p_pos)

    results = {
        "roc_auc": round(auc_point, 4),
        "buckets": [],
    }
    for b_idx, bucket in enumerate(buckets):
        results["buckets"].append({
            "bucket": b_idx,
            "size": bucket["size"],
            "score_min": round(bucket["score_min"], 4),
            "score_max": round(bucket["score_max"], 4),
            "n_sampled": int(n_pos[b_idx] + n_neg[b_idx]),
            "n_pos": int(n_pos[b_idx]),
            "n_neg": int(n_neg[b_idx]),
            "p_pos": round(float(p_pos[b_idx]), 4),
        })

    return results


# ─── PRF curve + Count Error Estimation ──────────────────────────────────────


def _bucket_pred_arrays(
    buckets: list[dict], predictions: np.ndarray
) -> list[np.ndarray]:
    """Return one sorted-ascending prediction array per bucket."""
    out = []
    for b in buckets:
        idx = np.asarray(b["indices"], dtype=int)
        out.append(np.sort(predictions[idx]))
    return out


def _candidate_thresholds(
    buckets: list[dict], predictions: np.ndarray, n_points: int = 200
) -> np.ndarray:
    """Choose threshold sweep points over the bucketed (unlabelled) pool.

    Quantile-spaced over the unlabelled prediction distribution so the curve
    has dense resolution where most patches live, plus the global min/max so
    the curve always extends to the trivial endpoints (recall=1 and recall=0).
    """
    all_preds = np.concatenate([
        np.asarray(b["indices"], dtype=int) for b in buckets
    ])
    pool = predictions[all_preds]
    if len(pool) == 0:
        return np.array([])
    qs = np.linspace(0.0, 1.0, n_points)
    ts = np.quantile(pool, qs)
    ts = np.unique(np.concatenate([[pool.min() - 1e-6], ts, [pool.max() + 1e-6]]))
    return ts


def compute_prf_curve(
    buckets: list[dict],
    eval_samples: list[list[str]],
    eval_labels: dict[str, int],
    predictions: np.ndarray,
    n_points: int = 200,
) -> list[dict]:
    """Estimate precision/recall/F1 as a function of logit threshold.

    Uses the within-bucket apportionment described in Navine et al. (2024)
    and the SeaSwipe phase2a quality-estimation script. For each candidate
    threshold t:

        est_TP(t) = Σ_b P(+|b) · |{i in bucket b : pred_i ≥ t}|
        est_FP(t) = Σ_b (1-P(+|b)) · |{i in bucket b : pred_i ≥ t}|
        est_FN(t) = Σ_b P(+|b) · |{i in bucket b : pred_i < t}|
        precision = est_TP / (est_TP + est_FP)
        recall    = est_TP / (est_TP + est_FN)
        f1        = 2·P·R / (P+R)

    Returns one dict per candidate threshold (sorted ascending).
    """
    p_pos, _, _ = _p_pos_per_bucket(buckets, eval_samples, eval_labels)
    bucket_sorted = _bucket_pred_arrays(buckets, predictions)
    thresholds = _candidate_thresholds(buckets, predictions, n_points)
    if len(thresholds) == 0:
        return []

    # est_total_pos is constant in t (whole pool).
    sizes = np.array([b["size"] for b in buckets], dtype=float)
    est_total_pos = float((p_pos * sizes).sum())

    curve = []
    for t in thresholds:
        est_tp = 0.0
        est_fp = 0.0
        for b_idx, arr in enumerate(bucket_sorted):
            n_above = int(arr.size - np.searchsorted(arr, t, side="left"))
            est_tp += p_pos[b_idx] * n_above
            est_fp += (1.0 - p_pos[b_idx]) * n_above
        est_fn = max(est_total_pos - est_tp, 0.0)
        denom_p = est_tp + est_fp
        denom_r = est_tp + est_fn
        precision = est_tp / denom_p if denom_p > 0 else 0.0
        recall = est_tp / denom_r if denom_r > 0 else 0.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        curve.append({
            "threshold": round(float(t), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "est_tp": round(float(est_tp), 4),
            "est_fp": round(float(est_fp), 4),
            "est_fn": round(float(est_fn), 4),
        })
    return curve


def best_f1_threshold(prf_curve: list[dict]) -> float | None:
    """Return the threshold that maximises F1 on the curve, or None if empty."""
    if not prf_curve:
        return None
    best = max(prf_curve, key=lambda p: p["f1"])
    return float(best["threshold"])


def compute_count_error_estimation(
    buckets: list[dict],
    eval_samples: list[list[str]],
    eval_labels: dict[str, int],
    predictions: np.ndarray,
    filename_to_idx: dict[str, int],
    threshold: float,
) -> dict:
    """Estimate TP / FP / FN counts on the full unlabelled pool at `threshold`.

    Designed to be readable to ecologists with no ML background.

      - **False Positives.** Take the 6×40 = 240 annotated patches whose
        logit is at or above `threshold`. The fraction of those marked
        negative is the observed FPR among the classifier's high-confidence
        predictions. Multiplying that fraction by the *total* number of
        patches above threshold extrapolates the FP count over the full
        unlabelled pool.

      - **True Positives.** est_TP = (1 - observed_FPR) · n_above_threshold,
        matching the SeaSwipe phase2a paper companion.

      - **False Negatives.** Within-bucket apportionment: for each bucket b,
        a fraction P(+|b) of its patches are positive; the ones below
        threshold are missed detections. Summing across buckets:
        est_FN = Σ_b P(+|b) · |{i in bucket b : pred_i < threshold}|

    Returns counts (rounded), percentages of the total estimated organism
    population (TP + FP + FN), and diagnostic counts so the UI can show
    "we used X annotations above threshold, Y of them were false".
    """
    p_pos, _, _ = _p_pos_per_bucket(buckets, eval_samples, eval_labels)
    bucket_sorted = _bucket_pred_arrays(buckets, predictions)

    # FN: within-bucket apportionment of P(+|b) onto patches below threshold.
    est_fn = 0.0
    n_below_per_bucket = []
    n_above_per_bucket = []
    for b_idx, arr in enumerate(bucket_sorted):
        cut = int(np.searchsorted(arr, threshold, side="left"))
        n_below = cut
        n_above = int(arr.size - cut)
        n_below_per_bucket.append(n_below)
        n_above_per_bucket.append(n_above)
        est_fn += float(p_pos[b_idx]) * n_below

    n_above_pool = sum(n_above_per_bucket)

    # Observed FPR on the 240 annotations restricted to logit ≥ threshold.
    n_annotated_above = 0
    n_annotated_above_neg = 0
    n_annotated_above_pos = 0
    for b_idx, samples in enumerate(eval_samples):
        for fn in samples:
            label = eval_labels.get(fn)
            if label is None:
                continue
            idx = filename_to_idx.get(fn)
            if idx is None:
                continue
            if predictions[idx] >= threshold:
                n_annotated_above += 1
                if label == 0:
                    n_annotated_above_neg += 1
                else:
                    n_annotated_above_pos += 1

    if n_annotated_above > 0:
        observed_fpr = n_annotated_above_neg / n_annotated_above
    else:
        observed_fpr = float("nan")

    if n_above_pool == 0 or np.isnan(observed_fpr):
        est_fp = float("nan")
        est_tp = float("nan")
    else:
        est_fp = observed_fpr * n_above_pool
        est_tp = (1.0 - observed_fpr) * n_above_pool

    # Total denominator for the percentages is TP + FP + FN: the estimated
    # *organism population* (everything the model got right, plus the
    # mistakes that matter to an ecological count). This is intentional and
    # different from classical accuracy (which would also include TN); the
    # UI surfaces "% of estimated organisms" so TN is irrelevant.
    # When FP is unknown (no annotations above threshold) we still report
    # FN since it is computed independently of the above-threshold pool.
    if np.isnan(est_tp) or np.isnan(est_fp):
        total = float("nan")
        tp_pct = fp_pct = fn_pct = float("nan")
    else:
        total = est_tp + est_fp + est_fn
        if total > 0:
            tp_pct = 100.0 * est_tp / total
            fp_pct = 100.0 * est_fp / total
            fn_pct = 100.0 * est_fn / total
        else:
            tp_pct = fp_pct = fn_pct = 0.0

    def _round_or_nan(x: float) -> float | None:
        return None if isinstance(x, float) and np.isnan(x) else round(float(x), 4)

    return {
        "threshold": round(float(threshold), 4),
        "n_above_threshold": int(n_above_pool),
        "n_annotated_above_threshold": int(n_annotated_above),
        "n_annotated_above_false_positive": int(n_annotated_above_neg),
        "n_annotated_above_true_positive": int(n_annotated_above_pos),
        "observed_fpr": _round_or_nan(observed_fpr),
        "est_true_positives": _round_or_nan(est_tp),
        "est_false_positives": _round_or_nan(est_fp),
        "est_false_negatives": round(float(est_fn), 4),
        "est_total_organisms": _round_or_nan(total),
        "tp_pct": _round_or_nan(tp_pct),
        "fp_pct": _round_or_nan(fp_pct),
        "fn_pct": _round_or_nan(fn_pct),
    }
