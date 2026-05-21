# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.11 conda env: `conda create -n AcneDetectionAnalysis python=3.11 -y && conda activate AcneDetectionAnalysis`
- `pip install -r requirements.txt` (only pins `opencv-python` and `ultralytics`; multitask work additionally needs `torch`, `transformers`, `Pillow`, `python-dotenv`).
- DINOv3 weights are gated on HuggingFace; the multitask trainer reads `HF_TOKEN` from a local `.env` (see `scripts/train_multitask.py:17`).

## Two parallel pipelines

This repo holds two independent approaches against the same lesion data. They share datasets but no code.

### 1. YOLO detector (mature, on `main`)

Trained, evaluated, and used for inference via the `ultralytics` package. Used scripts:

```bash
python scripts/train.py --model-name yolov8n --epochs 300 --patience 50 --dataset acne-dataset
python scripts/evaluate.py --model-name yolov8n
python scripts/inference.py --model-name yolov8n --image-path ./test_images/ --confidence 0.255
```

Important path convention: `train.py` writes weights to `Acne_Detection/<name>/weights/best.pt` (via `project=` arg to `model.train`), but Ultralytics nests this under `runs/detect/`. Both `evaluate.py` and `inference.py` hard-code the resolved path as `./runs/detect/Acne_Detection/<model-name>/weights/best.pt` — keep this in sync if you rename the project or move the output directory. Training defaults (`imgsz=1280`, `batch=2`, `workers=2`) are tuned to avoid RAM OOM on the dev machine, not for speed.

The `.pt` files at the repo root (`yolov8n.pt`, `yolo26{n,s,m,l}.pt`) are **pretrained backbones** that `YOLO("<name>.pt")` loads as a starting point — they are not trained checkpoints.

### 2. Multi-task DINOv3 classifier (experimental, current branch `train/ensemble`)

Severity classification (3 classes) + auxiliary lesion-density head, both driven from a shared frozen DINOv3 backbone.

```bash
python scripts/train_multitask.py --dataset-name acne04-dataset-512 --epochs 15 --batch-size 4
```

Module layout (`src/` is added to `sys.path` by the training script):

- `src/MultiTaskModule.py` — `MultiTaskDinoV3` wraps a HF `AutoModel` backbone. CLS token → linear `cls_head`; patch tokens → `DensityHead` (Conv decoder upsampled to `density_map_size`, default 56). Backbone is frozen by default and forced into `eval()` even during training (see `MultiTaskClassifier.py:295`).
- `src/MultiTaskAcneDataset.py` — reads YOLO label `.txt` files, converts box centers to a Gaussian density map, and derives the severity label from lesion count via `_detections_to_severity`: **0–5 = mild (0), 6–20 = moderate (1), >20 = severe (2)**. This rule is the only place severity is defined; treat it as the source of truth.
- `src/MultiTaskClassifier.py` — trainer wrapper. Composite loss = `LabelDistributionLoss` (KL to Gaussian-smoothed one-hot, encodes ordinality) + `density_loss_weight * MSE(density)` + `count_loss_weight * SmoothL1(sum(density), true_count)`. Weights default to `100.0` / `0.05` so the three terms land on the same order — re-tune if you change `density_map_size` or `sigma`, since the Gaussian normalizer scales with `1/sigma²`.

**Known broken import**: `scripts/train_multitask.py:14` imports `from MultiTask import MultiTaskImageClassifier`, but the class lives in `src/MultiTaskClassifier.py`. The script won't run until this is fixed to `from MultiTaskClassifier import MultiTaskImageClassifier`. This is in-progress work on `train/ensemble`.

Checkpoints save to `./saved_models/<model_name_with_slashes_replaced>_multitask_best.pt`. The trainer auto-resumes from this file if it exists (see `MultiTaskClassifier.py:104`).

## Datasets

Three sibling dataset folders, all YOLO format (`{split}/images/*.jpg` + `{split}/labels/*.txt` with normalized `cls cx cy w h`). Each has its own `data.yaml`:

- `acne-dataset/` — Kaggle "Acne Dataset in YOLOv8 Format", single class `Acne`, variable-resolution images. Uses an **absolute** `path:` in `data.yaml` — you must edit it after cloning.
- `acne04-dataset/` — Roboflow ACNE04, single class named `fore` (not `Acne`). Uses relative `path: ../`.
- `acne04-dataset-512/` — produced by `scripts/letterbox_dataset.py` from `acne04-dataset`. Images are letterbox-padded to 512×512 and labels are remapped to the new canvas. Use this for the multi-task pipeline so HF processors don't distort the box→density alignment.

To regenerate the 512 variant:

```bash
python scripts/letterbox_dataset.py --src acne04-dataset --dst acne04-dataset-512 --size 512
```

All three dataset directories plus `runs/`, `test_images/`, `*.pt`, and `*.zip` are gitignored.

## Single-source utilities

- `scripts/visualize.py` — dev-only helper. Hard-coded to `acne04-dataset-512` and writes `check_*.jpg` to CWD; edit the path at the bottom of the file.
- `test.log` — append-only log written by `evaluate.py`.
- `README.md` results table only covers the YOLO models against `acne-dataset` (Kaggle), not ACNE04.
