"""
Patch extraction — detect organisms in frames using a YOLO detector and crop
each detection into a padded patch.

Pure functions: no HTTP, no database, no task-queue dependencies. Writes one
JPEG per detection plus a COCO-format ``detections_coco.json`` so the source
frame and bounding box can be recovered later (used by the frontend lightbox).
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2

from src.backend.settings import (
    YOLO_CONFIDENCE,
    YOLO_IOU,
    YOLO_IMAGE_SIZE,
    PATCH_PADDING,
)

logger = logging.getLogger(__name__)


def extract_patches(
    frame_paths: list[Path],
    model_path: Path,
    patches_dir: Path,
    coco_path: Path,
    conf: float = YOLO_CONFIDENCE,
    iou: float = YOLO_IOU,
    imgsz: int = YOLO_IMAGE_SIZE,
    padding: float = PATCH_PADDING,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[dict, int]:
    """Run YOLO on each frame, save crops as patches, return COCO-formatted results.

    Each patch is named ``{frame_stem}_patch{annotation_id}.jpg``; the annotation
    id is the COCO annotation id so the lightbox can map a patch back to its
    source frame and bbox.

    Returns:
        (coco_dict, total_patches)
    """
    from ultralytics import YOLO

    patches_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLO model from %s", model_path)
    model = YOLO(str(model_path))

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": int(cid) + 1, "name": name}
            for cid, name in model.names.items()
        ],
    }
    annotation_id = 1
    total_patches = 0

    for i, frame_path in enumerate(frame_paths, start=1):
        if progress_callback is not None:
            progress_callback(i)
        image = cv2.imread(str(frame_path))
        if image is None:
            logger.warning("Could not read frame: %s", frame_path)
            continue

        img_h, img_w = image.shape[:2]
        image_id = i

        coco["images"].append({
            "id": image_id,
            "file_name": frame_path.name,
            "width": img_w,
            "height": img_h,
        })

        results = model(str(frame_path), iou=iou, augment=True, conf=conf, imgsz=imgsz)

        if not results or len(results[0].boxes) == 0:
            if i % 100 == 0:
                logger.info(
                    "Processed %d / %d frames (%d patches so far)",
                    i, len(frame_paths), total_patches,
                )
            continue

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu())
            category_id = int(box.cls[0].cpu()) + 1

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "score": score,
                "iscrowd": 0,
            })

            # Crop patch with padding (PATCH_PADDING of the bbox size per side).
            pad_x = bbox_w * padding
            pad_y = bbox_h * padding

            x1_padded = max(0, int(x1 - pad_x))
            y1_padded = max(0, int(y1 - pad_y))
            x2_padded = min(img_w, int(x2 + pad_x))
            y2_padded = min(img_h, int(y2 + pad_y))

            patch = image[y1_padded:y2_padded, x1_padded:x2_padded]

            patch_filename = f"{frame_path.stem}_patch{annotation_id}.jpg"
            cv2.imwrite(str(patches_dir / patch_filename), patch)
            total_patches += 1
            annotation_id += 1

        if i % 100 == 0:
            logger.info(
                "Processed %d / %d frames (%d patches so far)",
                i, len(frame_paths), total_patches,
            )

    coco_path.parent.mkdir(parents=True, exist_ok=True)
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    logger.info(
        "Done. %d patches from %d frames. COCO saved to %s",
        total_patches, len(frame_paths), coco_path,
    )

    return coco, total_patches


def discover_frames(frames_dir: Path) -> list[Path]:
    """Return all frame image files under ``frames_dir`` (sorted, recursive)."""
    from src.backend.pipeline.embedding import IMAGE_EXTENSIONS

    return sorted(
        f for f in frames_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
