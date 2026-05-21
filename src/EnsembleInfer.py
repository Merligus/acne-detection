"""
Ensemble inference: combine a trained MultiTaskImageClassifier with a trained
Ultralytics YOLO detector.

Rationale: the multi-task density head learns severity well but counts poorly
(stuck near zero at high LR — see sweep results). The YOLO detector counts well
(it's a proper detector) but doesn't directly output a severity class. Wire
them together so each model contributes the signal it's actually good at.

Severity fusion: convert the YOLO box count into a severity distribution by
placing a discrete Gaussian on the bucket the count falls into, then
linearly blend with the classifier's softmax:

    fused = alpha * classifier_probs + (1 - alpha) * yolo_severity_probs

`alpha=0.5` weights them equally. `alpha=1.0` recovers the classifier;
`alpha=0.0` recovers the YOLO-derived severity.
"""

import math
from typing import Dict, List, Optional, Tuple

from PIL import Image

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from MultiTaskClassifier import MultiTaskImageClassifier


SEVERITY_THRESHOLDS = (5, 20)  # 0..5 mild, 6..20 moderate, >20 severe


def count_to_severity_bucket(count: int) -> int:
    if count <= SEVERITY_THRESHOLDS[0]:
        return 0
    if count <= SEVERITY_THRESHOLDS[1]:
        return 1
    return 2


def count_to_severity_probs(count: int, num_classes: int = 3, sigma: float = 0.6) -> torch.Tensor:
    bucket = count_to_severity_bucket(count)
    idx = torch.arange(num_classes, dtype=torch.float32)
    diffs = idx - float(bucket)
    log_probs = -(diffs ** 2) / (2.0 * sigma ** 2)
    return torch.softmax(log_probs, dim=0)


class MultiTaskYOLOEnsemble:
    """
    Wrap a MultiTaskImageClassifier and an Ultralytics YOLO model. The
    classifier is loaded from a multi-task checkpoint dir; the YOLO is loaded
    from a `best.pt` path produced by `scripts/train.py`.

    Args:
        mt_ckpt_path: full path to a multi-task `.pt` checkpoint.
        mt_model_name: the HF model name used when training, e.g.
            'facebook/dinov3-vitb16-pretrain-lvd1689m'. Used to rebuild the
            image processor and backbone shell before loading state dict.
        hf_token: HF access token (DINOv3 weights are gated).
        yolo_weights_path: full path to a YOLO `best.pt`.
        yolo_conf: confidence threshold for YOLO detections (count = boxes
            above this threshold).
        alpha: severity-fusion weight on the classifier (0..1).
        sigma_count: spread of the count-derived severity distribution.
    """

    def __init__(
        self,
        mt_ckpt_path: str,
        mt_model_name: str,
        hf_token: str,
        yolo_weights_path: str,
        yolo_conf: float = 0.25,
        alpha: float = 0.5,
        sigma_count: float = 0.6,
        num_classes: int = 3,
        classes: Optional[List[int]] = None,
    ):
        self.alpha = alpha
        self.sigma_count = sigma_count
        self.num_classes = num_classes
        self.yolo_conf = yolo_conf

        self.mt = MultiTaskImageClassifier(
            model_name=mt_model_name,
            token=hf_token,
            num_classes=num_classes,
            classes=classes,
            ckpt_path=mt_ckpt_path,
            no_resume=False,
        )
        self.yolo = YOLO(yolo_weights_path)

    # ------------------------------------------------------------------
    # Per-image prediction
    # ------------------------------------------------------------------

    def predict_image(self, image_path: str) -> Dict:
        """
        Run both models on a single image. Returns a dict with all the
        intermediate signals plus the fused severity.
        """
        image = Image.open(image_path).convert("RGB")

        # Multi-task classifier branch.
        self.mt.model.eval()
        with torch.no_grad():
            inputs = self.mt.image_processor(images=image, return_tensors="pt").to(self.mt.device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits, density_pred = self.mt.model(inputs["pixel_values"])
            classifier_probs = torch.softmax(logits, dim=-1).squeeze(0).float().cpu()
            mt_count = float(density_pred.flatten(1).sum(dim=1).item())

        # YOLO branch. predict() is silent with verbose=False.
        yolo_results = self.yolo.predict(source=image_path, conf=self.yolo_conf, verbose=False)
        if yolo_results and yolo_results[0].boxes is not None:
            yolo_count = int(yolo_results[0].boxes.shape[0])
        else:
            yolo_count = 0

        yolo_probs = count_to_severity_probs(yolo_count, self.num_classes, self.sigma_count)

        fused_probs = self.alpha * classifier_probs + (1.0 - self.alpha) * yolo_probs
        fused_probs = fused_probs / fused_probs.sum()

        return {
            "classifier_probs": classifier_probs.tolist(),
            "classifier_severity": int(classifier_probs.argmax().item()),
            "classifier_count_estimate": mt_count,
            "yolo_count": yolo_count,
            "yolo_severity": count_to_severity_bucket(yolo_count),
            "yolo_probs": yolo_probs.tolist(),
            "fused_probs": fused_probs.tolist(),
            "fused_severity": int(fused_probs.argmax().item()),
        }

    # ------------------------------------------------------------------
    # Dataset evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _true_count_and_severity(label_path: str) -> Tuple[int, int]:
        n = 0
        try:
            with open(label_path, "r") as f:
                for line in f:
                    if len(line.strip().split()) >= 5:
                        n += 1
        except FileNotFoundError:
            n = 0
        return n, count_to_severity_bucket(n)

    def evaluate(self, samples: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        samples : list of (image_path, yolo_label_txt_path) tuples — same
            shape as MultiTaskAcneDataset's `samples`.

        Returns per-branch metrics so the user can see which signal each
        model contributes:
            cls_acc           — classifier-only severity accuracy
            yolo_sev_acc      — YOLO-count-derived severity accuracy
            ensemble_acc      — fused severity accuracy
            cls_count_mae     — classifier density-sum count MAE
            yolo_count_mae    — YOLO count MAE
        """
        n = len(samples)
        cls_hits = yolo_hits = ens_hits = 0
        cls_count_err = yolo_count_err = 0.0

        for img_path, lbl_path in samples:
            true_count, true_sev = self._true_count_and_severity(lbl_path)
            out = self.predict_image(img_path)

            cls_hits += int(out["classifier_severity"] == true_sev)
            yolo_hits += int(out["yolo_severity"] == true_sev)
            ens_hits += int(out["fused_severity"] == true_sev)

            cls_count_err += abs(out["classifier_count_estimate"] - true_count)
            yolo_count_err += abs(out["yolo_count"] - true_count)

        denom = max(n, 1)
        return {
            "n": n,
            "cls_acc": cls_hits / denom,
            "yolo_sev_acc": yolo_hits / denom,
            "ensemble_acc": ens_hits / denom,
            "cls_count_mae": cls_count_err / denom,
            "yolo_count_mae": yolo_count_err / denom,
        }
