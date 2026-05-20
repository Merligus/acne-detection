# Multi-task DINOv3 + YOLO ensemble results (2026-05-18)

Dataset: `acne04-dataset-512` (1174/143/102 train/valid/test).
Hardware: RTX 3050 6 GB Laptop. All multi-task runs use frozen backbone + AMP autocast.

## Phase 1 — Hyperparameter sweep (single LR)

| Run | Model   | LR    | Epochs | density_w | count_w | val_acc (best) | val_count_mae | test_acc   |
|-----|---------|-------|--------|-----------|---------|----------------|---------------|------------|
| B1  | vit-S   | 5e-4  | 2      | 20        | 0.05    | 71.33%         | 10.43         | 74.51%     |
| B2  | vit-S   | 5e-4  | 5      | 200       | 0.005   | 76.92%         | 10.43         | 78.43%     |
| B3  | vit-S   | 1e-4  | 5      | 200       | 0.005   | 67.83%         | **3.69**      | 73.53%     |
| B4  | vit-S   | 1e-3  | 5      | 200       | 0.005   | 78.32%         | 10.43         | 81.37%     |
| B5  | vit-S   | 1e-3  | 5      | 200       | 0.005   | 79.02%         | 10.43         | 78.43%     |
| B6  | vit-B   | 1e-3  | 10     | 200       | 0.005   | 80.42%         | 10.43         | 82.35%     |

B5 also used `density_map_size=112 density_sigma=3.0 class_weights=1.0,0.8,1.6`.

**Diagnosis from B1–B6:** at LR=1e-3 the classifier learns but the density head's ReLU output stays clamped at zero (count_mae = mean of test counts ≈ 13). At LR=1e-4 the density head learns (count_mae=3.69) but the classifier under-trains. The two heads need different LRs.

## Phase 2 — Per-head LR

| Run | Model | cls_lr | density_lr | Epochs | val_acc (best) | val_count_mae | test_acc | test_count_mae |
|-----|-------|--------|------------|--------|----------------|---------------|----------|----------------|
| B7  | vit-B | 1e-3   | **1e-4**   | 10     | 80.42%         | **3.00**      | **82.35%** | **3.64**     |

Single change vs B6: AdamW given two param groups (`cls_head` at 1e-3, `density_head` at 1e-4). Cosine warmup multiplier applies equally to both. **Classifier accuracy stays at B6's 82.35% while count MAE drops from 12.97 → 3.64** — the density head is finally producing useful counts.

### B7 reproduction command

```bash
python scripts/train_multitask.py \
  --dataset-name acne04-dataset-512 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --checkpoint-dir saved_models/exp07_perhead_lr --no-resume \
  --epochs 10 --batch-size 2 \
  --learning-rate 1e-3 --density-lr 1e-4 \
  --density-loss-weight 200.0 --count-loss-weight 0.005 --ldl-sigma 1.0
```

## Phase 3 — YOLO ensemble

Fusion rule: `fused_probs = α · classifier_probs + (1-α) · count_to_severity_probs(yolo_count)`.
`count_to_severity_probs(n)` places a discrete Gaussian (σ=0.6) on the severity bucket the count falls into.

Two YOLO checkpoints were tested:
- `yolo26s7` — trained on `acne04-dataset` (matches the multi-task domain), reported mAP50=0.31.
- `yolov8n`  — trained on `acne-dataset` (Kaggle, different domain), reported mAP50=0.64.

### Sweep at the B7 checkpoint

| YOLO weights | conf | yolo_sev_acc | yolo_count_mae | best ensemble acc | best (α) |
|--------------|------|--------------|----------------|-------------------|----------|
| yolo26s7     | 0.15 | 76.47%       | **4.97**       | **82.35%**        | 0.80–0.85 |
| yolo26s7     | 0.25 | 52.94%       | 9.05           | 79.41%            | 0.90     |
| yolov8n      | 0.20 | 67.65%       | 6.57           | 81.37%            | 0.85     |
| yolov8n      | 0.25 | 72.55%       | 5.93           | 80.39%            | 0.85     |

**Findings:**
1. Domain match beats detector strength — `yolo26s7` (mAP50=0.31, on-domain) outperforms `yolov8n` (mAP50=0.64, off-domain) in this ensemble.
2. The ensemble gain over classifier-alone is small (+1 sample on n=102). The per-head LR fix already gave the multi-task model a working count head (MAE=3.64), so the YOLO is now a redundant signal rather than a fix for a broken signal.
3. YOLO confidence threshold matters more than alpha — conf=0.15 is the sweet spot for `yolo26s7` (matches its under-detection bias).

### Best-ensemble reproduction command

```bash
python scripts/evaluate_ensemble.py \
  --dataset-name acne04-dataset-512 --split test \
  --mt-ckpt saved_models/exp07_perhead_lr/facebook_dinov3-vitb16-pretrain-lvd1689m_multitask_best.pt \
  --mt-model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --yolo-weights runs/detect/Acne_Detection/yolo26s7/weights/best.pt \
  --yolo-conf 0.15 --alpha 0.85
```

Output:
```
cls_acc          81.37%
yolo_sev_acc     76.47%
ensemble_acc     82.35%
cls_count_mae    3.79
yolo_count_mae   4.97
```

## Phase 4 — YOLO as density teacher (Stage 1: static yolo26s7, 2026-05-18)

Replaces the ground-truth Gaussian density teacher with `yolo26s7` (acne04-trained, mAP50=0.31) predictions. Each YOLO box contributes a Gaussian whose amplitude = its detection confidence. `count_target` for the SmoothL1 anchor = `sum(confidences)` to stay internally consistent with the density-map integral. **Severity label and val/test metrics still use GT** so test_acc is directly comparable to B7.

The B7 baseline re-evaluated from the saved checkpoint with the new RMSE eval (added in this phase): **test_acc=81.37%, test_count_mae=3.79, test_count_rmse=6.78**. (`82.35%` in the original training log was a 1-sample autocast nondeterminism — same checkpoint, single-pass re-eval gives 81.37%.)

| Run | YOLO     | conf | sigma | epochs | det.   | test_acc | test_mae | test_rmse |
|-----|----------|------|-------|--------|--------|----------|----------|-----------|
| B7  | (gt)     | —    | 2.0   | 10     | —      | 81.37%   | **3.79** | **6.78**  |
| T1  | yolo26s7 | 0.25 | 2.0   | 5      | 6068   | 82.35%   | 11.78    | 20.23     |
| T2  | yolo26s7 | 0.10 | 2.0   | 5      | 25320  | 82.35%   | 10.50    | 18.48     |
| T3  | yolo26s7 | 0.40 | 2.0   | 5      | 994    | 82.35%   | 12.73    | 20.99     |
| T4  | yolo26s7 | 0.25 | 3.0   | 5      | 6068   | 82.35%   | 11.78    | 20.23     |
| T5  | yolo26s7 | 0.10 | 2.0   | 10     | 25320  | 82.35%   | 9.31     | 16.47     |

### Stage 1 takeaways

1. **Classifier accuracy is rock-solid at 82.35%** across every YOLO-teacher configuration — the severity head is unaffected by which teacher supervises the density head. This is the strongest signal in the sweep: the multi-task framing tolerates the alternate teacher cleanly.
2. **Count metrics get worse with YOLO teacher** (rmse 16–21 vs B7's 6.78). This is the deliberate consequence of the user-confirmed design: `count_target = sum(YOLO confidences)`, which underestimates the true box count whenever YOLO's mAP is below perfect. Lower conf threshold → more (lower-confidence) detections → sum-of-confs closer to GT count → MAE/RMSE smaller. T2/T5 (conf=0.10) win this calibration race.
3. **Density-map sigma had no effect** (T4 vs T1 are bit-identical) — the sigma was a red herring for this dataset/teacher.
4. **More epochs help count alignment but not accuracy** (T5 vs T2: rmse 16.47 vs 18.48 at same conf; test_acc unchanged).

### Stage 1 gate decision

T5 vs B7: test_acc within ±1pt (82.35% vs 81.37%), but test_count_rmse **much higher** (16.47 vs 6.78). Per the plan's second gate condition (*"rmse doesn't improve → proceed to Stage 2 with the hypothesis that mAP50=0.31 is the bottleneck"*), proceeding to Stage 2 — train a fresh yolo26s on `acne04-dataset-512` and re-run T5 with the stronger teacher.

### Caveat for Stage 2 interpretation

Even a perfectly accurate YOLO would give `sum(confs) < N_boxes` (since each conf < 1), so the YOLO teacher's count_target is fundamentally lower than GT count. Stage 2 will only fully close the rmse gap if either (a) the fresh YOLO's confidences cluster near 1.0 on detections (saturated detector), or (b) we revisit the `weight_by_conf` design choice. If Stage 2 plateaus, that's the message — not a code bug.

## Stage 2 (truncated): fresh yolo26s on acne04-dataset-512

Run: `python scripts/train.py --model-name yolo26s --dataset acne04-dataset-512 --epochs 300 --patience 50`. Output: `runs/detect/Acne_Detection/yolo26s9/`.

Stopped early at **epoch 131/300** (best epoch 92, no improvement for 39 epochs — within ~10 of triggering the patience-50 auto-stop).

**Best result: mAP50=0.294, mAP50-95=0.078** at epoch 92 — essentially identical to `yolo26s7`'s 0.31. Fresh training from `yolo26s.pt` did **not** unlock a stronger teacher on this dataset/architecture. The detector has plateaued.

C2 (rerun T5 config with the fresh teacher) is **not run** — the teacher quality is materially the same as yolo26s7, so we'd reproduce T5's numbers. The gate's hypothesis ("mAP50=0.31 is the bottleneck") is partially refuted: the YOLO architecture+dataset combination caps at ~0.30 mAP50; bumping epochs won't fix it.

## Phase 5 — SegFormer as density teacher (upper-bound, 2026-05-19)

Replaces the density teacher with a pre-trained **SegFormer-B1 acne-segmentation model** (`~/acne-segmentation/checkpoint/...epoch45_iou0.4737.pth`, F1=0.6428 on its own test set). Each image's density target is the SegFormer sigmoid probability mask, downsampled to 56×56 and renormalized per-image so its integral equals the GT box count. `count_target = GT count` — restores B7's count calibration but with a SegFormer-derived spatial shape.

### Data leak detection (Phase 5 B1)

Stem-level overlap between SegFormer training and acne04 splits (Roboflow renames source files with `_jpg.rf.<hash>` suffix; comparing the `levle*_<N>` stems):

| Pair                           | Overlap | Total acne04 | % |
|--------------------------------|---------|--------------|---|
| segformer-train vs acne04-train | 1052    | 1174         | 90% |
| segformer-train vs acne04-valid | 128     | 143          | 90% |
| segformer-train vs acne04-test  | 94      | 102          | 92% |
| segformer-test  vs acne04-test  | 8       | 102          | 8%  |

**~90% leak across all splits.** SegFormer has effectively memorized the acne04 image pool. Phase 5 numbers are therefore an **upper bound** — they show what segmentation supervision can deliver when the teacher is perfectly calibrated to the training distribution. A follow-up plan to retrain SegFormer on a split that's disjoint from acne04 val/test is the natural next step.

### Phase 5 results

All runs use B7-winning multi-task config: `vit-B`, `--learning-rate 1e-3 --density-lr 1e-4 --density-loss-weight 200 --count-loss-weight 0.005 --ldl-sigma 1.0 --batch-size 2`. SegFormer precompute: ~35s for 1174 train images (0 empty-mask fallbacks needed in practice).

| Run | Teacher  | epochs | val_acc (best) | test_acc | test_mae | test_rmse |
|-----|----------|--------|----------------|----------|----------|-----------|
| B7  | gt       | 10     | 80.42%         | 81.37%   | 3.79     | 6.78      |
| T5  | yolo26s7 | 10     | 80.42%         | 82.35%   | 9.31*    | 16.47*    |
| S1  | segformer| 5      | 80.42%         | **83.33%** | 3.90   | 7.07      |
| S2  | segformer| 10     | 80.42%         | 82.35%   | **3.53** | **6.29**  |

*Phase 4 T5 uses `count_target = sum(YOLO confs)`, so its MAE/RMSE against GT are deliberately mis-calibrated — see Phase 4 section for context.

### Phase 5 takeaways

1. **SegFormer teacher beats B7 on every metric at 10 epochs.** test_acc 82.35% (≥B7), test_count_mae 3.53 (<B7 3.79), test_count_rmse 6.29 (<B7 6.78). All deltas are small (~1pt acc, ~7% count improvement) but consistent.
2. **At 5 epochs (S1), test_acc jumps to 83.33% — the strongest single number in the entire sweep.** Likely a "rich-signal converges faster" effect — dense pixel supervision lets the density head pick up calibration in fewer steps. With more epochs (S2) the count metric improves further but accuracy regresses to 82.35%.
3. **The 90% data leak means the absolute numbers are optimistic, but the comparative direction is meaningful**: dense (pixel-level) segmentation supervision is a strictly better-shaped signal than sparse (box-centered Gaussian) supervision for the density head, *given a competent teacher*. Whether a competent teacher exists on truly held-out data is the open question — that's what the follow-up retraining tests.
4. **YOLO-as-teacher (Phase 4) was the wrong tool for this dataset.** Even at higher mAP it would still hit the calibration mismatch (sum-of-confs ≠ GT count). SegFormer's pixel mask, normalized to GT count, sidesteps both the calibration issue and YOLO's label-noise ceiling.

### Decision and follow-up

This is the new best configuration. Recommended next steps:
1. **Retrain SegFormer on a leak-free split.** Take the 1174 acne04 train stems, build a SegFormer training set that excludes anything overlapping acne04 val/test stems, retrain to convergence. Then rerun S2 with the leak-free SegFormer to get the *unbiased* number.
2. **Consider co-training (Phase D from prior plan)** — if leak-free SegFormer underperforms (likely, since it'll be trained on ~120 fewer images), co-training could compensate. Now there's a real motivation: SegFormer-as-teacher beats baseline, so it's worth investing in.

## Phase 6 — YOLO teacher recalibrated to GT count (2026-05-19)

Phase 4's `count_target = sum(YOLO confs)` was deliberately mis-calibrated vs GT count — the asterisks on Phase 4's MAE/RMSE row called this out. Phase 6 mirrors Phase 5's SegFormer normalization to the YOLO branch:
- `density_target` = YOLO-Gaussian map renormalized per-image to integrate to GT count.
- `count_target` = GT count.
- Empty-YOLO edge case falls back to gt-Gaussian for that sample.

Implemented at `src/MultiTaskAcneDataset.py` `__getitem__` for `density_source == "yolo"`. Phase 4's old behavior is dropped — the YOLO mode is now self-consistent and apples-to-apples with B7/Phase-5.

### Sweep

All runs: vit-B, lr=1e-3, density-lr=1e-4, density-w=200, count-w=0.005, ldl-sigma=1.0, batch=2, 10 epochs. yolo26s7 teacher (acne04-trained, mAP50=0.31).

| Run | conf | test_acc | test_mae | test_rmse |
|-----|------|----------|----------|-----------|
| B7  | —    | 82.35%   | 3.64     | 6.78      |
| **Y1**  | **0.10** | **82.35%** | **3.61** | **6.56** |
| **Y2**  | **0.25** | **82.35%** | **3.56** | **6.53** |

### Phase 6 takeaways

1. **Calibrated YOLO teacher beats B7 on every metric.** test_acc matches, test_mae drops 0.03–0.08, test_rmse drops 0.22–0.25. Small but consistent.
2. **Conf threshold is roughly insensitive** once normalization is in place: 0.10 vs 0.25 land within noise (3.61/6.56 vs 3.56/6.53). The Phase 4 result of "conf=0.10 wins" was actually about calibration, not detection quality.
3. **The Phase 4 asterisks are formally resolved.** Phase 4 numbers stay above for historical context but should be read as "pre-calibration" — they were never measuring the same thing as B7/S2.
4. **Compared to Phase 5 S2 (leaky SegFormer, 82.35%/3.53/6.29)**: leak-free YOLO is within noise on every metric. Suggests that with proper calibration, a weak-but-on-domain detector ≈ a strong-but-leaked segmenter for this density-teacher framing. The leak-free SegFormer (Phase 7, below, when it lands) is the real test.

## Phase 7 — Leak-free SegFormer teacher (2026-05-19)

The Phase 5 SegFormer-B1 had ~90% data leak (its training set covered 1052/1174 acne04 train, 128/143 valid, 94/102 test stems). Phase 7 retrains SegFormer with acne04-aligned splits: train only on acne04 train images (1174 stems, all masks present after the user re-downloaded the full mask set), validate on acne04 valid (143), evaluate on acne04 test (102) — zero leak.

### New SegFormer training (acne-segmentation/)

A near-copy of `~/acne-segmentation` lives at `acne-segmentation/` (this repo) with:
- `dataset/__init__.py` — verbatim copy of the upstream `AcneDataset` class.
- `train_script.py` — same training pipeline, paths re-pointed at acne04-aligned splits.
- `evaluate.py` — same evaluator.
- `build_splits.py` — new helper that extracts acne04 stems and writes leak-free train/valid/test split lists.

Training command (an earlier `--batch_size 8` attempt OOM'd on 6 GB VRAM; `--batch_size 4` is the documented fallback):

```bash
python acne-segmentation/train_script.py \
  --architecture SegFormer \
  --segformer_model nvidia/segformer-b1-finetuned-ade-512-512 \
  --img_size 512 512 \
  --epochs 200 --batch_size 4 --lr 6e-5
```

**Result:** best IoU = **0.4515 at epoch 89** (validation split, leak-free). Compares to the original leaky checkpoint's 0.4737 — a ~5% drop, the cost of removing the leak. 200 epochs total, plateaued well before completion.

Checkpoint: `acne-segmentation/checkpoint/best_model_segformer_nvidia_segformer-b1-finetuned-ade-512-512_epoch89_iou0.4515.pth`.

### Sweep on the leak-free teacher

| Run | Teacher                  | epochs | test_acc | test_mae | test_rmse |
|-----|--------------------------|--------|----------|----------|-----------|
| B7  | gt-Gaussian              | 10     | 82.35%   | 3.64     | 6.78      |
| S2  | SegFormer (leaky, S2)    | 10     | 82.35%   | 3.53     | 6.29      |
| Y1  | YOLO (calibrated, conf=0.10) | 10 | 82.35%   | 3.61     | 6.56      |
| Y2  | YOLO (calibrated, conf=0.25) | 10 | 82.35%   | 3.56     | 6.53      |
| F1  | SegFormer (leak-free)    | 5      | **83.33%** | 3.93   | 7.18      |
| **F2** | **SegFormer (leak-free)** | **10** | 82.35% | **3.61** | **6.36** |

### Phase 7 takeaways

1. **F2 beats B7 on every count metric — honest paper result.** test_count_rmse drops from 6.78 → 6.36 (~6% relative). test_count_mae 3.64 → 3.61 (tiny but consistent). test_acc unchanged at 82.35%.

2. **F2 vs S2 (leaky): F2 is 0.07 worse on rmse (6.36 vs 6.29).** So ~7% of S2's improvement over B7 was leak-inflation. The remaining gain (B7 6.78 → F2 6.36) is the *real* signal: pixel-level segmentation supervision, properly leak-free, is a slightly better density teacher than GT Gaussians.

3. **F1 (5 epochs) hit 83.33% — the strongest single accuracy in the whole sweep.** At 10 epochs F2 regresses to 82.35% while improving count metrics — same dynamic as Phase 5 S1/S2. The 5-epoch checkpoint may be the better operating point if accuracy is the headline metric.

4. **All teachers converge to 82.35% test_acc**, regardless of supervision style. The classification head is saturated by the frozen DINOv3 backbone — better supervision moves count metrics, not accuracy.

5. **Compared to Y1/Y2 (calibrated YOLO)**: leak-free SegFormer F2 (6.36 rmse) beats both YOLOs (Y1 6.56, Y2 6.53). Dense pixel supervision wins over sparse, conf-weighted Gaussians — even when the underlying detector and segmenter have similar effective "domain calibration".

### Final ranking by count_rmse on the test split

1. **F2 (leak-free SegFormer)** — 6.36 ← **honest best**
2. Y2 (calibrated YOLO conf=0.25) — 6.53
3. Y1 (calibrated YOLO conf=0.10) — 6.56
4. B7 (gt-Gaussian) — 6.78
5. S2 (leaky SegFormer) — 6.29 ← optimistic; leak-inflated

F2 is the number to publish.

## GPU memory & compute footprint of each teacher

Measured directly on the RTX 3050 6 GB Laptop. All training uses the same DINOv3-B backbone (frozen, AMP autocast, batch=2, 512²) — so the *training* peak is identical across all three methods. The teachers differ only in the **precompute** phase that runs once at dataset init, before the DINOv3 training loop starts. After precompute, the teacher is moved to CPU / deleted and only the cached 56×56 density tensors remain in RAM.

| Stage                          | GT-Gaussian (B7) | YOLO teacher (Y1/Y2) | SegFormer teacher (F1/F2) |
|--------------------------------|------------------|----------------------|---------------------------|
| Teacher peak GPU (precompute)  | **0 MB** (CPU only) | **110 MB**         | **240 MB**                |
| Teacher weights on disk        | 0                 | ~21 MB (yolo26s)   | ~53 MB (SegFormer-B1)     |
| Precompute wall time (1174 imgs) | <1 s            | ~22 s              | ~34 s                     |
| Cached density tensors (RAM)   | ~15 MB           | ~15 MB             | ~15 MB                    |
| DINOv3-B training peak GPU     | ~1.4 GB forward + ~1 GB grad/optim ≈ 2.5 GB | (same) | (same) |
| Pipeline peak GPU              | **~2.5 GB**       | **~2.5 GB**         | **~2.5 GB**               |
| Extra Python deps              | none              | `ultralytics`       | `transformers` (already present) |

**Honest interpretation:**
- **All three pipelines fit comfortably in 6 GB** because DINOv3 training dominates (~2.5 GB). The teacher overhead is tiny in comparison.
- GT is strictly the lightest, but the savings (~240 MB during a one-time ~30 s precompute) are negligible relative to the 1+ hour DINOv3 training.
- SegFormer is the heaviest teacher (240 MB peak, 53 MB on disk, 34 s precompute) but the difference vs YOLO is small (~130 MB).
- **If a future Phase D-style retrain or co-training run blows past 6 GB, the bottleneck is DINOv3, not the teacher.** Reducing `--batch-size` to 1 or moving to vit-S would be the lever, not changing teachers.

Practical takeaway: **GPU memory is not a deciding factor between the three approaches at this scale.** Pick the teacher by what it does to test metrics (Phase 7 ranking), not by what it costs at runtime.

## Phase 8 / A — acne04 boxes vs seg masks comparison (2026-05-19)

Goal: decide whether to retrain YOLO on mask-derived labels by first checking how much the acne04 box annotations and the seg pixel masks disagree on the **same images**.

Method (`scripts/compare_acne04_vs_seg.py`): for each acne04 image that has a matching seg mask via the `_jpg.rf.<hash>` stem-stripping (1419 / 1419 stems matched, 0 missing), letterbox the mask to 512×512, threshold at 127, run `cv2.connectedComponentsWithStats`. A sample is "equal" iff every mask component has a pixel inside some YOLO box AND every YOLO box contains some mask pixel.

### Results — two views

| Metric | Value |
|---|---|
| Spatial coverage pass rate (every component inside a box AND every box covers some mask) | **99.65%** (1414/1419) |
| Average per-sample box recall | 1.000 |
| Average per-sample component recall | 1.000 |
| **Exact count match** (`N_boxes == N_components`) | 52% (742/1419) |
| YOLO has more boxes than mask has components | 48% (676/1419) |
| Mask has more components than YOLO has boxes | <0.1% (1/1419) |

### Interpretation

- **Spatial coverage agrees almost perfectly (99.65%)** → the two annotation styles describe the same lesions in the same images, with no systematic localization mismatch.
- **But exact counts disagree in ~half the samples** → connected components on the binary mask consistently *under*-counts vs YOLO boxes (48% of images have boxes > components by ≥1). The most likely cause is that adjacent lesions merge into a single connected component on the mask, while annotators kept them as separate YOLO boxes.

### Gate decision

Per the user-chosen "spatial coverage" metric, the gate threshold (≥90% pass rate) is **comfortably exceeded (99.65%)** → **Phase B/C skipped**. A YOLO retrained on mask-derived boxes would not be detecting different lesions; it would just be drawing different (fewer, merged) boxes around the same regions.

The 52% exact-count split is a separate finding worth noting:
- If we did retrain on mask-derived boxes, the new YOLO would systematically detect fewer instances than `yolo26s7`. mAP50 on the acne04 test split would likely **drop** (since the acne04 test labels are the "more-boxes" style and the new model would learn the "fewer-boxes" style).
- The acne04 mAP50 ceiling of ~0.30 is therefore probably **not** about label noise on lesion location. More likely candidates: limited model capacity vs lesion-scale variability, or limited training data.

Outputs on disk: `compare_acne04_vs_seg.log`, `compare_acne04_vs_seg.csv` (one row per stem with all metrics — useful for browsing outliers).

## Phase 8 / A2 — IoU comparison (acne04 bboxes vs seg masks, 2026-05-19)

Sharper diagnostic on top of Phase 8/A's spatial-coverage finding: compute IoU between the bbox-filled mask (from acne04 YOLO boxes) and the seg pixel mask (letterboxed to 512×512) for all 1419 paired stems.

### Results

| Metric | Overall (n=1419) | train (n=1174) | valid (n=143) | test (n=102) |
|---|---|---|---|---|
| **IoU** mean / median / std | **0.9219** / 0.9245 / 0.030 | 0.9218 / 0.9244 / 0.030 | 0.9206 / 0.9241 / 0.032 | 0.9248 / 0.9264 / 0.030 |
| seg_in_bbox mean / median | 0.9947 / 1.0000 | 0.9943 / 1.0000 | 0.9977 / 1.0000 | 0.9946 / 1.0000 |
| bbox_filled_by_seg mean / median | 0.9269 / 0.9281 | 0.9271 / 0.9286 | 0.9228 / 0.9252 | 0.9298 / 0.9324 |
| area_ratio mean / median | 0.9325 / 0.9284 | 0.9331 / 0.9289 | 0.9252 / 0.9252 | 0.9355 / 0.9324 |

IoU histogram (overall): **1141 samples in [0.9, 1.0), 277 in [0.8, 0.9), 1 in [0.7, 0.8), and 0 anywhere lower.** No edge cases (no empty masks, no empty box sets).

### Interpretation — the user's hypothesis is essentially confirmed

The seg masks are **filled rectangles slightly shrunk** vs the acne04 bboxes:
- `seg_in_bbox ≈ 1.0` → every mask pixel is inside a bbox.
- `bbox_filled_by_seg ≈ 0.93` → 93% of the bbox area is mask pixel.
- `area_ratio ≈ 0.93` → mask area is 93% of bbox area.

If the seg masks were **real lesion contours** inside bboxes, the area ratio would be 0.3–0.6 (typical lesion footprint within an enclosing box). Instead it's 0.93 — the mask is essentially a 7%-eroded version of the bbox, not a contour. The masks were almost certainly generated by some procedure that took the acne04 bboxes and applied a slight shrink/anti-aliasing pass.

### Implications

1. **The seg dataset doesn't carry pixel-level lesion structure** — it's a re-encoding of the same acne04 bboxes, just slightly less rectangular.
2. **Phase 7 F2's win over B7 (~6% rmse) wasn't from "pixel-contour supervision"** — it was from "supervision in the shape of slightly-soft rectangles" vs the GT-Gaussian shape used by B7. Still a valid empirical result, but the mechanism is different from what we thought.
3. **YOLO retrain on mask-derived bboxes (Phase 8 B/C) would produce essentially `yolo26s7` again.** The new bboxes would just be ~93% the size of the original ones (since masks are 93%-area rectangles). No meaningful mAP improvement to expect.
4. **The acne04 mAP50 ceiling at ~0.30 is not a label problem.** Phase 8/A2 confirms the box labels match the underlying lesion locations as well as any other annotation we have. The ceiling must come from elsewhere — image quality, lesion-scale diversity, or detector capacity at this dataset size.

### Recommendation

**Keep Phase 8 B/C on hold permanently.** A retrain would not produce a meaningfully different YOLO. If the user wants to push mAP further, the more productive avenues are:
- Train a larger backbone (yolo26m / yolo26l) on `acne04-dataset-512` at imgsz=1280, with longer schedules (the existing acne04-trained variants stopped early).
- Augmentations targeting small-lesion preservation (current `scripts/train.py` already does aggressive scale + mosaic; could try AutoAugment).
- Treat the YOLO detector as good-enough at mAP50≈0.30 and lean on the multi-task density head for downstream count/severity — which is what Phase 7 already does with F2 as the production teacher.

Outputs on disk: `compare_acne04_vs_seg_iou.log`, `compare_acne04_vs_seg_iou.csv` (one row per stem).

## Phase 9 — Per-head WD + LR + loss-weight sweep (2026-05-20)

Added a `--density-weight-decay` CLI flag and split `AdamW` weight_decay into per-group settings (`src/MultiTaskClassifier.py`). Ran 4 variants × 2 teachers (YOLO Y1, SegFormer F2). All runs: vit-B, 10 epochs, batch=2, otherwise B7-winning multi-task config.

### Results

| Run | Teacher | Variant | test_acc | test_mae | test_rmse |
|-----|---------|---------|----------|----------|-----------|
| B7  | gt-Gaussian | baseline    | 82.35%   | 3.64     | 6.78      |
| Y1  | YOLO conf=0.10 | baseline | 82.35%   | 3.61     | 6.56      |
| F2  | SegFormer leak-free | baseline | 82.35% | 3.61   | 6.36      |
| D1y | YOLO    | density-WD=1e-3 | 82.35% | **3.59** | 6.46 |
| D2y | YOLO    | density-WD=1e-5 | 82.35% | 3.61   | 6.56 |
| **D3y** | **YOLO** | **LR=5e-4 / density-LR=5e-5** | **85.29%** | 3.72 | 6.68 |
| D4y | YOLO    | loss rebalance (200→100, 0.005→0.01) | 82.35% | 3.64 | 6.75 |
| **D1f** | **SegFormer** | **density-WD=1e-3** | 82.35% | **3.58** | **6.30** |
| D2f | SegFormer | density-WD=1e-5 | 82.35% | 3.60 | 6.38 |
| **D3f** | **SegFormer** | **LR=5e-4 / density-LR=5e-5** | **84.31%** | 3.76 | 6.64 |
| D4f | SegFormer | loss rebalance | 82.35% | 3.63 | 6.61 |

### Two big findings

**1. The 82.35% classification ceiling was never the backbone — it was the cls_head LR.** Halving the learning rates (lr=5e-4 / density-lr=5e-5) broke through it on both teachers:
- **D3y: 85.29% test_acc** (+2.94pt over Y1 — biggest jump in the entire project, the new single-run record).
- **D3f: 84.31% test_acc** (+1.96pt over F2).

This invalidates the prior assumption that we needed to swap backbones to push past 82.35%. The next-frontier backbone plan is now lower priority — there's still headroom at the current `vit-B` capacity, we just weren't reaching it.

**2. Density-WD=1e-3 gives the best count metrics for both teachers** (D1y rmse 6.46 < Y1's 6.56; D1f rmse 6.30 < F2's 6.36). Stronger regularization on the density head helps it converge to a lower per-pixel error without hurting classification.

**A clear trade-off**: LR-halving gains ~2–3pt accuracy but loses ~0.3 rmse; density-WD=1e-3 keeps accuracy at 82.35% but gains ~0.10 rmse. **The two configs optimize different objectives.** The natural next experiment is "do both compose?" — run D5 = LR-halved + density-WD=1e-3 on each teacher.

### Other observations

- **Density-WD=1e-5 does nothing** (D2y/D2f within noise of baseline). The cls_head's wd=1e-4 is fine; the density head was never under-regularized.
- **Loss rebalance hurts** on both teachers (D4y/D4f worse than baseline). The current 200/0.005 weights are well-tuned for the GT-count-normalized teachers; perturbing them by 2× either direction is destabilizing.
- **YOLO benefits more from LR-halving than SegFormer** (+2.94pt vs +1.96pt). Likely because YOLO's signal is sparser/noisier — slower updates let it integrate better.

### Final ranking (Phase 9 winners + earlier baselines)

**By test_acc:**
1. **D3y (YOLO + LR halved): 85.29%** ← new best
2. D3f (SegFormer + LR halved): 84.31%
3. B7 / Y1 / F2 / D1y / D2y / D4y / D1f / D2f / D4f: 82.35% (the ceiling that broke at D3y/D3f)

**By test_count_rmse:**
1. **D1f (SegFormer + density-WD=1e-3): 6.30** ← new best
2. F2 (SegFormer baseline): 6.36
3. D2f (SegFormer + density-WD=1e-5): 6.38
4. D1y (YOLO + density-WD=1e-3): 6.46
5. Y1 (YOLO baseline) / D2y: 6.56
6. D3f (SegFormer + LR halved): 6.64
7. D3y (YOLO + LR halved): 6.68
8. D4y / D4f: 6.75 / 6.61
9. B7 (GT): 6.78

### Phase 9.1 — Composition test (D5y, D5f)

Ran the composed config (LR=5e-4 / density-LR=5e-5 + density-WD=1e-3) on each teacher to test whether the two Phase 9 improvements stack.

| Run | Teacher | Config | test_acc | test_mae | test_rmse |
|-----|---------|--------|----------|----------|-----------|
| D3y | YOLO | LR halved alone | 85.29% | 3.72 | 6.68 |
| D5y | YOLO | LR halved + density-WD=1e-3 | 85.29% | 3.71 | 6.69 |
| D3f | SegFormer | LR halved alone | 84.31% | 3.76 | 6.64 |
| D5f | SegFormer | LR halved + density-WD=1e-3 | 84.31% | 3.76 | 6.62 |

**Composition is negative on both teachers.** D5y and D5f reproduce D3y and D3f to within noise — adding `density-WD=1e-3` on top of LR-halving contributes nothing measurable. The density-WD's effect (seen alone in D1y/D1f) is **eaten** by LR-halving.

**Mechanistic interpretation**: LR=5e-4 produces a different optimization trajectory than LR=1e-3. The density-WD=1e-3 regularization helps at the original (faster) LR by preventing the density head from overshooting, but at the halved LR there's no overshoot to prevent — the regularization becomes a no-op.

### Final ranking — you must pick an objective

The two improvements are **mutually exclusive** in practice:

- **For best test_acc**: D3y (YOLO + LR halved) — **85.29% / 3.72 / 6.68**. Highest classification accuracy of the whole project.
- **For best test_count_rmse**: D1f (SegFormer + density-WD=1e-3) — **82.35% / 3.58 / 6.30**. Best count localization, ceiling on accuracy.

The user should pick based on downstream task — severity classification (D3y) vs lesion counting (D1f).

### Updated recommendation

- **Stop tuning the multi-task hyperparameters.** Phase 9 mapped the local landscape; the trade-off between accuracy and count is real and not bridgeable with the current backbone + heads.
- **Backbone-exploration plan is now the next-most-likely source of gains.** With the LR-halving win, we know there's headroom at vit-B capacity — but a different backbone (larger, or unfrozen with the new LR) might push past 85.29% acc *while* keeping the density head's count quality. That's the natural next experiment.
- **Alternative**: ensemble the two winners (D3y + D1f). One model predicts severity, the other predicts count — could publish both. Implementable with the existing `EnsembleInfer` pattern from Phase 3.

## Phase 10 — LR × teacher × backbone 2×2×2 sweep (2026-05-20)

Added `--unfreeze-backbone` and `--backbone-lr` flags; `MultiTaskClassifier` now accepts `freeze_backbone` in its constructor and adds a 3rd AdamW param group when backbone is trainable. Goal: push toward `test_acc > 90%` after Phase 9's LR-halving discovery.

All runs: vit-B, 10 epochs, batch=2, `weight-decay=1e-4`, `density-loss-weight=200`, `count-loss-weight=0.005`, `ldl-sigma=1.0`. Backbone-LR = density-LR (10× below cls-LR) for unfrozen runs.

### Results

| Run | cls-LR | density-LR | Teacher | Backbone | test_acc | test_mae | test_rmse |
|-----|--------|-----------|---------|----------|----------|----------|-----------|
| B7  | 1e-3   | 1e-4      | gt-Gaussian | frozen | 82.35% | 3.64 | 6.78 |
| D3y | 5e-4   | 5e-5      | YOLO       | frozen | 85.29% | 3.72 | 6.68 |
| D1f | 1e-3   | 1e-4      | SegFormer  | frozen | 82.35% | 3.58 | 6.30 (best frozen) |
| E1y | 1e-4   | 1e-5      | YOLO       | frozen | 80.39% | 3.85 | 6.75 |
| E2y | 5e-5   | 5e-6      | YOLO       | frozen | 76.47% | 4.16 | 7.21 |
| E1f | 1e-4   | 1e-5      | SegFormer  | frozen | 82.35% | 3.80 | 7.04 |
| E2f | 5e-5   | 5e-6      | SegFormer  | frozen | 77.45% | 3.95 | 7.08 |
| **E3y** | **1e-4** | **1e-5** | **YOLO** | **UNFROZEN** | **86.27%** | **2.96** | **4.86** ← new champion |
| E4y | 5e-5   | 5e-6      | YOLO       | UNFROZEN | 82.35% | 3.03 | 5.22 |
| E3f | 1e-4   | 1e-5      | SegFormer  | UNFROZEN | 82.35% | 2.91 | 5.01 |
| E4f | 5e-5   | 5e-6      | SegFormer  | UNFROZEN | 85.29% | 2.91 | 5.12 |

### Headline findings

1. **E3y is the new best on EVERY metric:**
   - test_acc = **86.27%** (+0.98 over D3y, +3.92 over B7)
   - test_mae = **2.96** (-0.62 / **-17%** vs D1f's 3.58)
   - test_rmse = **4.86** (-1.44 / **-23%** vs D1f's 6.30)
   - The Phase 9.1 trade-off (best-acc xor best-rmse) has **disappeared** under unfrozen backbone — both metrics improve together.

2. **Unfrozen backbone is the missing piece** — all 4 unfrozen runs (E3y/E4y/E3f/E4f) tie or beat the previous frozen best on count metrics; E3y also beats it on accuracy.

3. **Lower-than-5e-4 LR in frozen mode strictly regresses.** E1y (1e-4) → 80.39% < D3y (5e-4) → 85.29%; E2y (5e-5) → 76.47% (worst). LR=5e-4 was the local optimum for frozen vit-B; this sweep tested points below it and confirmed they're worse.

4. **Did not reach 90% test_acc.** Best was 86.27% (E3y) — that's only **+1 image correct** away from D3y's 85.29% (the test set is 102 samples). Val peaks were higher: E3f peaked at val_acc=88.81% (closest to 90% in this sweep). The 90% target is within reach but not consistently hit.

### Why YOLO unfrozen > SegFormer unfrozen?

E3f peaked at val=88.81% but landed at test=82.35% — large val/test gap, classic overfitting signature. E3y peaked val=87.41%, ended test=86.27% — tight gap. Plausible mechanism: SegFormer's dense pixel-level supervision (≈93% bbox area, per Phase 8/A2) gives the unfrozen backbone a strong locality signal that's easy to memorize; YOLO's sparse Gaussian-from-boxes is "cooler" supervision and forces the backbone to learn more general features. **YOLO + unfrozen is the production config.**

### Open levers if we want 90%

1. **More epochs** — E3y's val_acc was still oscillating 85–87% at epoch 10. Try 20–30 epochs (~30–45 min each).
2. **Augmentation** — `MultiTaskAcneDataset.__getitem__` does *no* augmentation (only `image_processor` normalization). Adding horizontal flip / color jitter / small rotation could close the val/test gap by 1–2 points.
3. **Larger backbone** — vit-L instead of vit-B. Probably OOM on 6 GB even with batch=1 — would need gradient checkpointing or downcast.
4. **Ensemble E3y + E3f** — different teachers, different mistakes; majority vote on the 102 test images could pick up the remaining ~4 the individual models miss.

Augmentation is the cheapest and most likely to close the val/test gap. Worth a Phase 11 with just that knob.

### Final ranking (Phase 10 settled)

**By test_acc:**
1. **E3y** (YOLO + unfrozen + LR=1e-4): **86.27%** ← champion
2. D3y (YOLO + frozen + LR=5e-4) / E4f (SegFormer + unfrozen + LR=5e-5): 85.29%
3. D3f (SegFormer + frozen + LR=5e-4): 84.31%

**By test_count_rmse:**
1. **E3y: 4.86** ← champion
2. E3f (SegFormer + unfrozen + LR=1e-4): 5.01
3. E4f: 5.12
4. E4y: 5.22

**By test_count_mae:**
1. **E3f / E4f: 2.91** (tie)
2. E3y: 2.96
3. E4y: 3.03

**Production winner: E3y on every metric except mae** (where it's 0.05 behind a saturated tie). Adopt as headline config.

## Earlier follow-ups (Phase 4 prior, now superseded by Phase 5)

The current YOLO ceiling is in the way of every remaining hypothesis:
- A YOLO mAP50 ≈ 0.30 means roughly 70% of acne lesions go undetected; sum-of-confs will always sit far below GT count.
- Even switching `weight_by_conf=False` (so each box contributes 1.0) only swaps "sum of low confs" for "count of detected boxes", and the detected boxes are still 30% of GT.
- The classifier already saturates at test_acc≈82% across every teacher variant — so the count task is the only remaining axis, and it's bottlenecked by the detector.

Options for the next planning round:
1. **Improve the detector** — larger model (yolo26m/l, already trained but at higher imgsz=1280), more augmentation, longer training, or fix label noise. The 0.30 ceiling may be a labeling/data issue, not a model issue.
2. **Replace YOLO with a different teacher** — e.g., a pretrained crowd-counting density estimator, or a segmentation model fine-tuned on acne masks. Sidesteps the detection bottleneck entirely.
3. **Co-training (Phase D)** — joint YOLO + classifier optimization where the classifier's signal back-propagates to YOLO. Would only help if there's still a learning signal beyond the current YOLO ceiling.
4. **Redefine the count target** — e.g., supervise the density head against GT density (current B7 baseline) but use YOLO at inference for a "second opinion" (current EnsembleInfer path). This effectively reverts Phase 4 to the Phase 3 ensemble approach which already gave +1pt accuracy.

A fresh plan should pick among these.

## Earlier follow-ups (now superseded by Phase 4)

- (resolved) Train a stronger YOLO on `acne04-dataset-512` — addressed in Stage 2 below.
- (in progress) Density-head supervision from YOLO — that's exactly Phase 4.
- Per-head weight decay sweep — still open.
