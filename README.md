# SeaSwipe

**Agile-modelling image classification for ecological imagery.**

SeaSwipe turns a folder of underwater frames into a per-taxon detector with a
few dozen clicks. It runs an object detector over your frames, embeds every
detected patch with a self-supervised vision transformer, and then lets you
train a lightweight linear classifier for one taxon at a time through an
interactive *agile-modelling* loop: pick a seed, label its nearest neighbours,
train, inspect the prediction histogram, label the patches the model is least
sure about, and repeat. It then estimates the classifier's ROC-AUC,
precision/recall, and count error from a single label-efficient annotation pass,
and exports the per-patch predictions as CSV.

This is the open-source companion to the SeaSwipe paper. The method is an
adaptation of **Agile Modeling** (Stretcu et al. 2023) and its bioacoustic
incarnation (Dumoulin et al. 2025), with the label-efficient quality-estimation
procedures of Navine et al. (2024).

---

## How it works

```
 frames/                                  (your input: a folder of images)
   │
   │  python seaswipe_preprocess.py  ──  foundation detector + image embedding
   ▼
 seaswipe_output/
   ├── patches/            cropped detections (+ detections_coco.json)
   ├── embeddings.npz      L2-normalised embedding vectors, one per patch
   └── manifest.json
   │
   │  python seaswipe_analysis.py  ──  interactive web workspace
   ▼
 seaswipe_output/analyses/<taxon>/
   ├── <taxon>_labels.json
   ├── <taxon>_classifier.npz       trained linear probe (weights, bias)
   ├── <taxon>_predictions.csv      per-patch logit + probability
   └── <taxon>_evaluation.json      ROC-AUC, PRF curve, count error
```

The two stages are decoupled: preprocessing is a one-off batch job (slow, GPU);
analysis is interactive and runs against the precomputed embeddings (fast, CPU).

---

## Installation

Requires **Python 3.10+** and **Node 18+** (to build the frontend once).

```bash
# 1. Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Build the web frontend (one-off)
cd frontend
npm install
npm run build
cd ..
```

For image embedding, we use DINOv3 by default. DINOv3 weights are downloaded automatically by `transformers` on first use. The
default model (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is gated on Hugging
Face — accept the model terms and log in (`huggingface-cli login`, or set
`HF_TOKEN`) before running preprocessing. A GPU is strongly recommended for
preprocessing; everything else runs comfortably on CPU.

---

## Usage

### 1. Preprocess — frames → patches + embeddings

```bash
python seaswipe_preprocess.py \
    --frames   /path/to/frames \
    --detector /path/to/detector.pt \
    --output   ./seaswipe_output
```

- `--frames` — a folder of frame images (searched recursively).
- `--detector` — a YOLO `.pt` detector that proposes candidate organisms.
- `--output` — where to write patches, embeddings, and the manifest.

Useful options: `--embedding-model <id-or-path>`, `--device cuda|cpu`,
`--batch-size`, `--conf`, `--iou`, `--imgsz`, `--padding`, and `--skip-detection`
(re-embed existing patches without re-running the detector). See
`python seaswipe_preprocess.py --help`.

#### Choosing a foundation detector

You can pass any foundation detector using the  `--detector` flag. By default, SeaSwipe accepts detectors in YOLO `.pt` formats. A few good off-the-shelf options:


| Detector                                                                                | What it is                                                                                                                                                 | Best for                                                                                     | SeaSwipe compatibility                                                                                                                                                                      |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[MegaDetector](https://github.com/agentmorris/MegaDetector)**                         | A widely used detector that finds animals, people, and vehicles (it localises, it doesn't identify to species). Trained on millions of camera-trap images. | Above-water / shoreline wildlife frames, generic "find the animal" proposals.                | Drop-in. Download a YOLO `.pt` release and pass it to `--detector`.                                                                                                                         |
| **[Community Fish Detector](https://github.com/filippovarini/community-fish-detector)** | A single-class "fish" detector trained on a community dataset of >1.9M images across freshwater, marine, and lab environments.                             | Underwater fish frames — the closest match to SeaSwipe's typical use case.                   | Use the `cfd-yolov12x` YOLO `.pt` release directly with `--detector`. (RF-DETR releases are not YOLO `.pt` and need conversion.)                                                            |

Feel free to extend the code to support other formats, such as SAM3, VLMs etc. 

### 2. Analyse — one taxon at a time

```bash
python seaswipe_analysis.py --data ./seaswipe_output
# then open http://127.0.0.1:8080
```

The page asks for a **taxon name** and then opens the workspace:

- **Train.** Pick a seed patch (random sampling or filename search), label its
nearest neighbours as positive/negative, and click *Train Classifier*. Inspect
the prediction-logit histogram, then *Sample* the patches the model is least
certain about (Entropy Sampling at the decision boundary) and label those.
Repeat for 3–5 rounds (typically 50–200 labels total) until the histogram
cleanly separates the classes.
- **Evaluate.** Label 40 random patches in each of 6 score buckets (240 labels)
to estimate ROC-AUC, the precision/recall/F1 curve, and a count-error
breakdown (true / false positives / false negatives), all from that single
annotation pass.
- **Results.** Browse the predicted-positive patches at any confidence
threshold (the F1-optimal threshold is suggested automatically), click a patch
to see its source frame with the detection box, and export the predictions as
CSV or the full analysis as a ZIP.

Run **one analysis per taxon**. State is saved under
`seaswipe_output/analyses/<taxon>/`, so you can stop and relaunch to resume,
or switch taxon from the workspace header.

> **Single-user, single-session.** The analysis server holds one active analysis
> in memory and is not locked for concurrent writers. Don't open the same
> dataset from multiple tabs at once.

### Frontend development (optional)

Instead of building, you can run the Vite dev server with hot reload while the
API runs separately:

```bash
python seaswipe_analysis.py --data ./seaswipe_output   # API on :8080
cd frontend && npm run dev                              # UI on :3000 (proxies /api)
```

---

## Default hyperparameters

All values live in `[src/backend/settings.py](src/backend/settings.py)`. The
single random seed (42) is propagated to detector inference, classifier
training, and evaluation bucket sampling, so every analysis is reproducible from
the same labels.


| Stage                   | Parameter               | Default                                    |
| ----------------------- | ----------------------- | ------------------------------------------ |
| **Frame extraction**    | Sampling rate           | 1 frame per 20 s of video¹                 |
| **Foundation detector** | Patch padding           | 30% of bounding-box size                   |
| **Patch embedding**     | Backbone                | DINOv3 ViT-L/16 (pre-trained on LVD-1689M) |
|                         | Embedding dimension *D* | 1024                                       |
|                         | Batch size              | 64                                         |
|                         | Normalisation           | L2                                         |
| **Active learning**     | Sampling strategy       | Entropy Sampling                           |
|                         | Sample size *K*         | 10                                         |
| **Similarity search**   | Top similar samples     | 3⁄4 *K* (15)                               |
|                         | Top dissimilar samples  | 1⁄4 *K* (5)                                |
| **Classifier**          | Architecture            | Linear layer (*D* → 1), no hidden layers   |
|                         | Loss                    | Binary cross-entropy with logits           |
|                         | Optimiser               | Adam                                       |
|                         | Learning rate           | 1 × 10⁻³                                   |
|                         | Epochs                  | 128                                        |
|                         | Mini-batch size         | 32                                         |
| **ROC-AUC estimation**  | Number of buckets *B*   | 6                                          |
|                         | Labels per bucket *J*   | 40                                         |
| **Reproducibility**     | Random seed (global)    | 42                                         |


¹ This tool starts from a folder of frames and does **not** extract frames from
video; the cadence is recorded here for reproducibility of the published frame
set. For a binary linear probe, *Entropy Sampling* is implemented as sampling
the patches whose predicted logit is closest to 0 (the decision boundary), where
the Bernoulli entropy of `sigmoid(logit)` is maximal.

---

## Outputs

Each analysis writes the following under `seaswipe_output/analyses/<taxon>/`:


| File                           | Contents                                                                   |
| ------------------------------ | -------------------------------------------------------------------------- |
| `<taxon>_labels.json`          | The patches you labelled during training (positive/negative)               |
| `<taxon>_classifier.npz`       | Trained linear-probe `weights` and `bias`                                  |
| `<taxon>_predictions.csv`      | Per-patch `filename, logit, probability`                                   |
| `<taxon>_evaluation.json`      | Buckets, ROC-AUC, PRF curve, F1-optimal threshold, count error             |
| `training_rounds/round_*.json` | A snapshot of the labels used at each training round (for learning curves) |
| `analysis.json`                | The taxon name and run id                                                  |


---

## Project structure

```
seaswipe_preprocess.py        CLI: frames → patches + embeddings
seaswipe_analysis.py          CLI: launch the analysis web app
src/backend/
  settings.py                 all hyperparameters (single source of truth)
  state.py                    filesystem-backed workspace + analysis state
  main.py                     FastAPI app (serves the API and the built UI)
  models.py                   request schemas
  api/analysis.py             the analysis API
  cli/                        preprocess + analysis entry points
  pipeline/
    patch_extraction.py       YOLO detection + padded cropping (+ COCO)
    embedding.py              DINOv3 embedding, L2 normalisation
    agile_modeling.py         cosine search, linear-probe training, sampling
    evaluation.py             ROC-AUC, PRF curve, count-error estimation
frontend/                     Vite + React + TypeScript single page
```

---

## References

- Stretcu, O., Platzer, A., Moriarty, T., et al. (2023). *Agile Modeling: From
Concept to Classifier in Minutes.* arXiv:2306.02313.
- Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual Features without
Supervision.* TMLR. (DINOv3 follows the same family.)
- Navine, A.K., Denton, T., Weldy, M.J., Hart, P.J. (2024). *All Thresholds
Barred: Direct Estimation of Call Density in Bioacoustic Data.* Frontiers in
Bird Science, 3:1380636.
- Dumoulin, V., Stretcu, O., et al. (2025). *The Search for Squawk: Agile
Modeling in Bioacoustics.* arXiv:2505.03071.

---

## License

[MIT](LICENSE) © 2026 OceanX.

## Citation

If you use SeaSwipe in your research, please cite the accompanying paper. *(BibTeX
to be added on publication.)*