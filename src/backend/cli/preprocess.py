"""Preprocessing CLI: a folder of frames -> patches + embeddings.

Runs the YOLO detector over every frame, crops each detection into a padded
patch (writing a COCO ``detections_coco.json`` so the source frame can be
recovered later), then embeds every patch with DINOv3 and L2-normalises.

Produces a self-contained output directory consumed by ``seaswipe_analysis.py``:

    output/
    ├── manifest.json
    ├── embeddings.npz
    └── patches/
        ├── *.jpg
        └── detections_coco.json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.backend import settings
from src.backend.pipeline.patch_extraction import extract_patches, discover_frames
from src.backend.pipeline.embedding import compute_embeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("seaswipe.preprocess")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seaswipe_preprocess.py",
        description="Detect organisms in frames and embed the detected patches.",
    )
    p.add_argument("--frames", required=True, type=Path,
                   help="Folder of input frame images (searched recursively).")
    p.add_argument("--detector", required=True, type=Path,
                   help="Path to the YOLO detector weights (.pt).")
    p.add_argument("--output", type=Path, default=Path("./seaswipe_output"),
                   help="Output directory (default: ./seaswipe_output).")
    p.add_argument("--embedding-model", default=settings.EMBEDDING_MODEL,
                   help="DINOv3 model id or local path "
                        f"(default: {settings.EMBEDDING_MODEL}).")
    p.add_argument("--device", default=None,
                   help="Torch device, e.g. cuda / cpu (default: auto-detect).")
    p.add_argument("--batch-size", type=int, default=settings.EMBEDDING_BATCH_SIZE,
                   help=f"Embedding batch size (default: {settings.EMBEDDING_BATCH_SIZE}).")
    p.add_argument("--conf", type=float, default=settings.YOLO_CONFIDENCE,
                   help=f"Detector confidence threshold (default: {settings.YOLO_CONFIDENCE}).")
    p.add_argument("--iou", type=float, default=settings.YOLO_IOU,
                   help=f"Detector NMS IoU (default: {settings.YOLO_IOU}).")
    p.add_argument("--imgsz", type=int, default=settings.YOLO_IMAGE_SIZE,
                   help=f"Detector inference image size (default: {settings.YOLO_IMAGE_SIZE}).")
    p.add_argument("--padding", type=float, default=settings.PATCH_PADDING,
                   help=f"Patch padding as a fraction of the bbox (default: {settings.PATCH_PADDING}).")
    p.add_argument("--skip-detection", action="store_true",
                   help="Reuse existing patches in <output>/patches and only re-embed.")
    return p


def run(args: argparse.Namespace) -> int:
    frames_dir = args.frames.resolve()
    output_dir = args.output.resolve()
    patches_dir = output_dir / "patches"
    coco_path = patches_dir / "detections_coco.json"
    embeddings_path = output_dir / "embeddings.npz"

    if not frames_dir.exists():
        logger.error("Frames directory not found: %s", frames_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    n_patches = 0

    if args.skip_detection:
        if not patches_dir.exists():
            logger.error("--skip-detection set but no patches at %s", patches_dir)
            return 1
        logger.info("Skipping detection; re-embedding existing patches in %s", patches_dir)
    else:
        if not args.detector.exists():
            logger.error("Detector weights not found: %s", args.detector)
            return 1
        frame_paths = discover_frames(frames_dir)
        n_frames = len(frame_paths)
        if n_frames == 0:
            logger.error("No frame images found under %s", frames_dir)
            return 1
        logger.info("Detecting organisms in %d frames ...", n_frames)
        _, n_patches = extract_patches(
            frame_paths=frame_paths,
            model_path=args.detector,
            patches_dir=patches_dir,
            coco_path=coco_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            padding=args.padding,
        )
        if n_patches == 0:
            logger.error("No detections produced — nothing to embed. "
                         "Try lowering --conf.")
            return 1

    logger.info("Embedding patches with %s ...", args.embedding_model)
    n_embedded, dim = compute_embeddings(
        patches_dir=patches_dir,
        output_path=embeddings_path,
        model_id=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    if n_embedded == 0:
        logger.error("No embeddings produced.")
        return 1

    manifest = {
        "frames_dir": str(frames_dir),
        "detector": str(args.detector.resolve()) if not args.skip_detection else None,
        "embedding_model": args.embedding_model,
        "embedding_backbone": settings._model_id_to_backbone_version(args.embedding_model),
        "n_frames": n_frames,
        "n_patches": n_patches if not args.skip_detection else n_embedded,
        "n_embeddings": n_embedded,
        "embedding_dim": dim,
        "padding": args.padding,
        "yolo": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz},
        "created": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Done. %d embeddings (dim=%d) written to %s",
                n_embedded, dim, embeddings_path)
    print()
    print(f"Preprocessing complete: {output_dir}")
    print(f"  patches:    {n_embedded}")
    print(f"  embeddings: {embeddings_path}")
    print()
    print(f"Next: python seaswipe_analysis.py --data {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
