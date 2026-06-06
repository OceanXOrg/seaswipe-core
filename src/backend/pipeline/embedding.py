"""
Patch embedding — compute DINOv3 embeddings for detected organism patches.

Pure functions: no HTTP, no database, no task-queue dependencies. Each patch is
encoded with DINOv3 (ViT-L/16, LVD-1689M) into a 1024-d vector, then
L2-normalised so cosine similarity downstream reduces to a dot product.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.backend.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    _model_id_to_backbone_version,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def model_id_to_backbone_version(model_id: str) -> str:
    """Convert a HuggingFace model id / local path to a short backbone string.

    e.g. 'facebook/dinov3-vitl16-pretrain-lvd1689m' -> 'dinov3-vitl16'.
    """
    return _model_id_to_backbone_version(model_id)


def discover_images(patches_dir: Path) -> tuple[list[str], list[Path]]:
    """Find all image files under the patches directory (recursively).

    Only the bare basename is stored in the embeddings .npz, which is also how
    patches are indexed downstream, so basenames must be unique.
    """
    files = sorted(
        f for f in patches_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    filenames = [f.name for f in files]
    return filenames, files


@torch.inference_mode()
def embed_batch(
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of PIL images and return a (B, D) numpy array."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy()


def compute_embeddings(
    patches_dir: Path,
    output_path: Path,
    model_id: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    device: str | None = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[int, int]:
    """Compute DINOv3 embeddings for all patches and save to .npz.

    Returns:
        (n_embedded, embedding_dim)
    """
    if not patches_dir.exists():
        logger.error("Patches directory not found: %s", patches_dir)
        return 0, 0

    device_obj = (
        torch.device(device) if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("Scanning patches in %s ...", patches_dir)
    filenames, filepaths = discover_images(patches_dir)
    logger.info("Found %d images", len(filenames))

    if not filenames:
        logger.warning("No images found. Skipping.")
        return 0, 0

    # Embeddings are indexed by bare basename downstream (filename -> idx). A
    # duplicate basename would silently collapse two patches into one entry and
    # corrupt every analysis built on this .npz, so refuse to proceed.
    if len(set(filenames)) != len(filenames):
        from collections import Counter
        dups = [name for name, count in Counter(filenames).items() if count > 1]
        sample = ", ".join(dups[:5])
        more = "" if len(dups) <= 5 else f" (+{len(dups) - 5} more)"
        raise ValueError(
            f"Duplicate patch basenames found under {patches_dir}: {sample}{more}. "
            "Patch filenames must be unique. Check that frame filenames are unique."
        )

    logger.info("Loading DINOv3 model (%s) on %s ...", model_id, device_obj)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, device_map=str(device_obj))
    model.eval()

    logger.info("Computing embeddings (batch_size=%d) ...", batch_size)
    all_embeddings = []

    for i in range(0, len(filepaths), batch_size):
        batch_paths = filepaths[i : i + batch_size]
        images = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception as e:
                logger.warning("Could not load %s: %s — using black placeholder", fp.name, e)
                img = Image.new("RGB", (224, 224))
            images.append(img)

        embeddings = embed_batch(model, processor, images, device_obj)
        all_embeddings.append(embeddings)

        done = min(i + batch_size, len(filepaths))
        if progress_callback is not None:
            progress_callback(done)
        logger.info("  Embedded %d / %d", done, len(filepaths))

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # L2-normalise for cosine similarity downstream.
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / np.maximum(norms, 1e-8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Saving %d embeddings (dim=%d) to %s",
        all_embeddings.shape[0], all_embeddings.shape[1], output_path,
    )
    # Write atomically: a crash mid-savez would otherwise leave a half-written
    # .npz that subsequent loads accept then fail on. (np.savez auto-appends
    # .npz, so the tmp path must already end in .npz.)
    tmp_path = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez(
        tmp_path,
        embeddings=all_embeddings.astype(np.float32),
        filenames=np.array(filenames),
    )
    tmp_path.replace(output_path)
    logger.info("Done.")

    return int(all_embeddings.shape[0]), int(all_embeddings.shape[1])
