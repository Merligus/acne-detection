# Acne Detection Analysis

Multi-task DINOv3 classifier that predicts **severity grade (mild / moderate / severe)** and an **auxiliary lesion density / count** from a single face image. The pipeline started as a YOLO detector (baseline) and grew into a multi-task model with several alternative density teachers (ground-truth Gaussians, calibrated YOLO predictions, leak-free SegFormer-B1 mask supervision).

Dataset: [ACNE04](https://universe.roboflow.com/andrei-dore-5lz05/acne04/dataset/1), letterboxed to 512×512 (`acne04-dataset-512/`).

## Headline ranking (test split, n=102, vit-B backbone)

All numbers are re-evaluated from the **saved best-val checkpoint** (the one we'd actually ship). See `saved_models/SWEEP_RESULTS.md` for the full sweep history.

| # | Run | Teacher | Backbone | Notes | test_acc | test_mae | test_rmse |
|---|-----|---------|----------|-------|----------|----------|-----------|
| 🥇 | **F1y + TTA** | YOLO conf=0.10 | unfrozen | + augmentation, 20 epochs, 4-variant TTA at inference | **88.24%** | **3.18** | **5.63** |
| 🥈 | **E3y** | YOLO conf=0.10 | unfrozen | 10 epochs, no augmentation | 86.27% | **2.96** | **4.86** ← best count |
| 🥉 | **E3f + TTA** | SegFormer leak-free | unfrozen | 10 epochs + 4-variant TTA | 86.27% | 2.95 | 5.25 |

**Pareto picks**: F1y wins accuracy, E3y wins count metrics, E3f gives an alternate teacher option.

## Install

```bash
conda create -n AcneDetectionAnalysis python=3.11 -y
conda activate AcneDetectionAnalysis

# Install torch first per the official selector for your CUDA version:
#   https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

`.env` (project root) needs an HF token for the gated DINOv3 + SegFormer models:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

## Datasets

- `acne04-dataset-512/` — letterboxed to 512×512. Used by every multi-task run. Generated via `python scripts/letterbox_dataset.py --src acne04-dataset --dst acne04-dataset-512 --size 512`.
- `~/acne-segmentation/data/{JPEGImages,mask}/` — pixel-mask supervision used by the SegFormer teacher. See `acne-segmentation/build_splits.py` for the leak-free split lists.

## Train the top 3

All three reuse the multi-task winning hyperparameters (`density-loss-weight=200`, `count-loss-weight=0.005`, `ldl-sigma=1.0`, vit-B, batch=2). The `train.log` ends up at the project root; move it into the checkpoint dir after each run.

### 🥇 F1y — augmented YOLO teacher (88.24% test_acc)

```bash
python scripts/train_multitask.py \
  --dataset-name acne04-dataset-512 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --checkpoint-dir saved_models/F1y --no-resume \
  --epochs 20 --batch-size 2 --eval-every-steps 300 \
  --learning-rate 1e-4 --density-lr 1e-5 \
  --weight-decay 1e-4 \
  --density-loss-weight 200.0 --count-loss-weight 0.005 --ldl-sigma 1.0 \
  --density-source yolo \
  --yolo-teacher-weights runs/detect/Acne_Detection/yolo26s7/weights/best.pt \
  --yolo-teacher-conf 0.10 \
  --unfreeze-backbone --augment
```

At inference, run TTA for the headline number:

```bash
python scripts/evaluate_tta.py \
  --dataset-name acne04-dataset-512 --split test \
  --mt-ckpt saved_models/F1y/facebook_dinov3-vitb16-pretrain-lvd1689m_multitask_best.pt
```

### 🥈 E3y — YOLO teacher, no augmentation (best count: rmse 4.86)

```bash
python scripts/train_multitask.py \
  --dataset-name acne04-dataset-512 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --checkpoint-dir saved_models/E3y --no-resume \
  --epochs 10 --batch-size 2 --eval-every-steps 300 \
  --learning-rate 1e-4 --density-lr 1e-5 \
  --weight-decay 1e-4 \
  --density-loss-weight 200.0 --count-loss-weight 0.005 --ldl-sigma 1.0 \
  --density-source yolo \
  --yolo-teacher-weights runs/detect/Acne_Detection/yolo26s7/weights/best.pt \
  --yolo-teacher-conf 0.10 \
  --unfreeze-backbone
```

Do **not** run TTA on this checkpoint — TTA hurts E3y because it never saw spatial augmentation during training.

### 🥉 E3f — SegFormer teacher + TTA (86.27% with TTA, best mae 2.89)

First train (or reuse) the leak-free SegFormer-B1 teacher:

```bash
# Build acne04-aligned split lists for the segmentation training set
python acne-segmentation/build_splits.py

# Retrain SegFormer-B1 on those splits — ~2 hr at batch=4 on 6 GB
python acne-segmentation/train_script.py \
  --architecture SegFormer \
  --segformer_model nvidia/segformer-b1-finetuned-ade-512-512 \
  --img_size 512 512 \
  --epochs 200 --batch_size 4 --lr 6e-5
```

Then the multi-task run:

```bash
python scripts/train_multitask.py \
  --dataset-name acne04-dataset-512 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --checkpoint-dir saved_models/E3f --no-resume \
  --epochs 10 --batch-size 2 --eval-every-steps 300 \
  --learning-rate 1e-4 --density-lr 1e-5 \
  --weight-decay 1e-4 \
  --density-loss-weight 200.0 --count-loss-weight 0.005 --ldl-sigma 1.0 \
  --density-source segformer \
  --segformer-weights acne-segmentation/checkpoint/<best_segformer_epochXX_iouY.YYYY.pth> \
  --unfreeze-backbone
```

Apply TTA at inference (E3f gains +2.94% accuracy from TTA):

```bash
python scripts/evaluate_tta.py \
  --dataset-name acne04-dataset-512 --split test \
  --mt-ckpt saved_models/E3f/facebook_dinov3-vitb16-pretrain-lvd1689m_multitask_best.pt
```

## YOLO detector (baseline, used as teacher in F1y/E3y)

```bash
python scripts/train.py --model-name yolo26s --dataset acne04-dataset-512 --epochs 300 --patience 50 --imgsz 512
python scripts/evaluate.py --model-name yolo26s
python scripts/inference.py --model-name yolo26s --image-path test_images/ --confidence 0.10
```

`yolo26s7` (mAP50=0.31) is the teacher used by F1y/E3y. See `scripts/compare_acne04_vs_seg_iou.py` for the bbox-vs-mask diagnostic that explains why the acne04 ceiling sits near mAP50=0.30.

## Other utilities

| Script | Purpose |
|--------|---------|
| `scripts/letterbox_dataset.py` | Letterbox a YOLO dataset to a fixed square size (used to build `acne04-dataset-512`). |
| `scripts/compare_acne04_vs_seg.py` / `compare_acne04_vs_seg_iou.py` | Phase 8 diagnostics — confirm acne04 boxes and seg masks describe the same lesions. |
| `scripts/visualize_segformer.py` | Render SegFormer mask overlays on a few sample images. |
| `acne-segmentation/evaluate.py` | IoU/F1 evaluator for the segmentation model. |
| `scripts/evaluate_ensemble.py` | Inference-time ensemble of multi-task classifier + YOLO detector (Phase 3 path). |

See `saved_models/SWEEP_RESULTS.md` for the full sweep history (11 phases, ~50 runs).
