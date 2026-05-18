# Multi-task DINOv3 sweep results (2026-05-18)

Dataset: `acne04-dataset-512` (1174/143/102 train/valid/test).
Hardware: RTX 3050 6 GB Laptop. All runs use frozen backbone, AMP autocast.

| Run | Model   | LR    | Epochs | density_w | count_w | val_acc (best) | val_count_mae | test_acc   |
|-----|---------|-------|--------|-----------|---------|----------------|---------------|------------|
| B1  | vit-S   | 5e-4  | 2      | 20        | 0.05    | 71.33%         | 10.43         | 74.51%     |
| B2  | vit-S   | 5e-4  | 5      | 200       | 0.005   | 76.92%         | 10.43         | 78.43%     |
| B3  | vit-S   | 1e-4  | 5      | 200       | 0.005   | 67.83%         | **3.69**      | 73.53%     |
| B4  | vit-S   | 1e-3  | 5      | 200       | 0.005   | 78.32%         | 10.43         | 81.37%     |
| B5  | vit-S   | 1e-3  | 5      | 200       | 0.005   | 79.02%         | 10.43         | 78.43%     |
| B6  | vit-B   | 1e-3  | 10     | 200       | 0.005   | **80.42%**     | 10.43         | **82.35%** |

B1 default loss weights, B5 also used `density_map_size=112 density_sigma=3.0 class_weights=1.0,0.8,1.6`.

## Winning config (B6)

```bash
python scripts/train_multitask.py \
  --dataset-name acne04-dataset-512 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --epochs 10 --batch-size 2 \
  --learning-rate 1e-3 \
  --density-loss-weight 200.0 \
  --count-loss-weight 0.005 \
  --ldl-sigma 1.0
```

## Open issue: density head doesn't learn at LR=1e-3

The density head only escaped the ReLU-zero-init at LR=1e-4 (B3 → count_mae 3.69), but at that LR the classifier suffers (test 73.53%). At LR=1e-3 the classifier is great (82.35%) but the density head stays dead (count_mae 10.43 = the mean of the val count distribution, i.e. the head outputs ~0 everywhere). Per-head learning rates or a different output activation (softplus / scaled sigmoid) would unblock this. Out of scope for this sweep; flagged for the YOLO-integration conversation.

## Class imbalance, density-map resolution

B5 explored both levers vs B4 baseline and found:
- Best val_acc nudged up (78.32% → 79.02%).
- test_acc dropped (81.37% → 78.43%).
- Effects are within noise. Not worth keeping for the final config.
