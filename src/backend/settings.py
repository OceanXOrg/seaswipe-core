"""Application settings — all configuration and default hyperparameters.

This is the single source of truth for the SeaSwipe Agile Modelling
hyperparameters reported in the paper. Every stochastic step reads
``RANDOM_SEED`` from here, so the whole pipeline is reproducible from one knob.
"""

import os
import re

# ── Agile Modelling ──────────────────────────────────────────────────────────

AGILE_MODELING_PARAMS = {
    "train_epochs": 128,
    "batch_size": 32,
    "learning_rate": 0.001,
    # Similarity search initialisation: with the default sample size K = 10,
    # 3/4 K = 10 nearest neighbours and 1/4 K = 5 dissimilar patches are shown.
    "initialisation_nearest_neighbours": 15,
    "initialisation_n_dissimilar": 5,
}

# Default active-learning sample size K. For a binary linear probe, sampling the
# patches whose predicted logit is closest to 0 (the decision boundary) is
# exactly Entropy Sampling — the Bernoulli entropy of sigmoid(logit) is maximal
# at logit = 0. So "Entropy Sampling, K = 10" is implemented by margin sampling
# at target_logit = 0 with n = 10.
DEFAULT_SAMPLE_SIZE = 10

EVALUATION_PARAMS = {
    "n_buckets": 6,        # B — logarithmic quantile buckets
    "labels_per_bucket": 40,  # J — labels per bucket (240 labels total)
}

# Single seed for every stochastic operation in the pipeline: linear-probe init
# and batch shuffling, and ROC-AUC bucket sampling. Fixing it here makes every
# analysis exactly reproducible from the same labels.json. Override per-call to
# ensemble or run a sensitivity analysis.
RANDOM_SEED = 42

# ── Detector (YOLO) ──────────────────────────────────────────────────────────

YOLO_CONFIDENCE = 0.2
YOLO_IOU = 0.3
YOLO_IMAGE_SIZE = 1280
PATCH_PADDING = 0.3  # 30% of the bounding-box size, padded on each side

# ── Patch embedding (DINOv3) ─────────────────────────────────────────────────

# DINOv3 ViT-L/16 pre-trained on LVD-1689M. Embedding dimension D = 1024.
EMBEDDING_MODEL = os.environ.get(
    "SEASWIPE_EMBEDDING_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m"
)
EMBEDDING_BATCH_SIZE = 64  # patches per forward pass
# Embeddings are L2-normalised so cosine similarity reduces to a dot product.


def _model_id_to_backbone_version(model_id: str) -> str:
    """e.g. 'facebook/dinov3-vitl16-pretrain-lvd1689m' -> 'dinov3-vitl16'.

    Falls back to the bare model name (or path basename) when the id does not
    match the expected DINOv3 naming scheme.
    """
    name = str(model_id).rstrip("/").split("/")[-1]
    match = re.match(r"(dinov\d+-vit\w+\d+)", name)
    return match.group(1) if match else name


EMBEDDING_BACKBONE_VERSION = _model_id_to_backbone_version(EMBEDDING_MODEL)

# ── Histogram ────────────────────────────────────────────────────────────────

HISTOGRAM_BINS = 60

# ── Confidence threshold ─────────────────────────────────────────────────────
# "High-confidence positive" prediction default (logit >= this). The Results tab
# overrides this with the F1-optimal threshold once an evaluation has been run.
DEFAULT_CONFIDENCE_THRESHOLD = 1.0
