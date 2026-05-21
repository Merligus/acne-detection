# ---------------------------------------------------------------------------
# Trainer wrapper -- mirrors ImageClassifier API
# ---------------------------------------------------------------------------

import os
import math
import json
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from MultiTaskModule import MultiTaskDinoV3


class LabelDistributionLoss(nn.Module):
    """
    KL divergence between predicted softmax and a discrete Gaussian centered
    on the true ordinal label. Encodes the inductive bias that off-by-one
    errors are less wrong than off-by-many errors, which plain CE ignores.

    sigma controls how much mass leaks to neighbors:
        0.5  -> sharp, close to one-hot (mild ordinal smoothing)
        1.0  -> moderate (good default for 3-4 ordinal classes)
        1.5  -> very smooth, lots of mass on neighbors

    class_weights (optional): per-class multiplier applied to the per-sample
    KL term before averaging. Useful for imbalanced datasets — pass a tensor
    of shape [num_classes].
    """

    def __init__(self, num_classes: int, sigma: float = 1.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.register_buffer(
            "class_idx",
            torch.arange(num_classes, dtype=torch.float32),
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def _build_targets(self, labels: torch.Tensor) -> torch.Tensor:
        labels_f = labels.float().unsqueeze(1)
        diffs = self.class_idx.unsqueeze(0) - labels_f
        log_probs = -(diffs**2) / (2.0 * self.sigma**2)
        return torch.softmax(log_probs, dim=1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target = self._build_targets(labels)
        log_pred = torch.log_softmax(logits, dim=1)
        if self.class_weights is None:
            return F.kl_div(log_pred, target, reduction="batchmean")
        per_sample = F.kl_div(log_pred, target, reduction="none").sum(dim=1)
        weights = self.class_weights[labels]
        return (per_sample * weights).mean()


class MultiTaskImageClassifier:
    """
    Same surface as the original ImageClassifier; the differences are:
      - the model returns (logits, density_map)
      - train/eval batches must include 'density_target' and 'count'
      - infer() also returns the integrated count

    Loss weights:
      density_loss_weight : MSE on density values is naturally tiny because
                            individual Gaussian peaks are ~1/(2*pi*sigma^2).
                            100.0 brings it onto the same order as CE.
      count_loss_weight   : Smooth-L1 on integrated count vs. true count.
                            Anchors the magnitude of the density head.
    Tune these by watching train_cls / train_dens / train_cnt in the logs --
    they should all be the same order of magnitude after a few hundred steps.
    """

    def __init__(
        self,
        model_name,
        token,
        num_classes,
        classes=None,
        ckpt_path=None,
        checkpoint_dir=None,
        logging=None,
        image_size: int = 512,
        density_map_size: int = 56,
        density_loss_weight: float = 20.0,
        count_loss_weight: float = 0.05,
        ldl_sigma: float = 1.0,
        class_weights=None,
        no_resume: bool = False,
        freeze_backbone: bool = True,
    ):
        self.model_name = model_name
        self.logging = logging
        self.num_classes = num_classes
        self.classes = classes
        self.image_size = image_size
        self.density_map_size = density_map_size
        self.density_loss_weight = density_loss_weight
        self.count_loss_weight = count_loss_weight
        self.ldl_sigma = ldl_sigma
        self.class_weights = class_weights

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._ckpt_filename = model_name.replace("/", "_") + "_multitask_best.pt"

        if ckpt_path is None and checkpoint_dir is not None and not no_resume:
            candidate = os.path.join(checkpoint_dir, self._ckpt_filename)
            if os.path.isfile(candidate):
                ckpt_path = candidate
                if logging:
                    logging.warning(f"Resuming from existing checkpoint: {candidate}")

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            ProcessorClass = getattr(
                transformers,
                ckpt["config"]["image_processor"]["image_processor_type"],
            )
            self.image_processor = ProcessorClass(**ckpt["config"]["image_processor"])
            self.backbone = AutoModel.from_config(AutoConfig.for_model(**ckpt["config"]["backbone"]))
            self.classes = ckpt["config"]["classes"]
            self.freeze_backbone = ckpt["config"].get("freeze_backbone", True)
            self.density_map_size = ckpt["config"].get("density_map_size", density_map_size)
            self.model = MultiTaskDinoV3(
                backbone=self.backbone,
                num_classes=len(self.classes),
                freeze_backbone=self.freeze_backbone,
                density_map_size=self.density_map_size,
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

            self.image_processor_config = ckpt["config"]["image_processor"]
            self.backbone_config = ckpt["config"]["backbone"]
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                token=token,
                do_center_crop=False,
                size={"height": self.image_size, "width": self.image_size},
            )
            self.backbone = AutoModel.from_pretrained(
                self.model_name,
                token=token,
            )
            self.image_processor_config = json.loads(self.image_processor.to_json_string())
            self.backbone_config = json.loads(
                AutoConfig.from_pretrained(
                    self.model_name,
                    token=token,
                ).to_json_string()
            )
            self.freeze_backbone = freeze_backbone
            self.model = MultiTaskDinoV3(
                self.backbone,
                self.num_classes,
                freeze_backbone=self.freeze_backbone,
                density_map_size=self.density_map_size,
            ).to(self.device)

    # ----- loss --------------------------------------------------------------

    def _multi_task_loss(
        self,
        logits,
        density_pred,
        labels,
        density_target,
        count_target,
        criterion_cls,
    ):
        loss_cls = criterion_cls(logits, labels)
        loss_density = F.mse_loss(density_pred, density_target)
        count_pred = density_pred.flatten(1).sum(dim=1)
        loss_count = F.smooth_l1_loss(count_pred, count_target)

        total = loss_cls + self.density_loss_weight * loss_density + self.count_loss_weight * loss_count
        return total, loss_cls.item(), loss_density.item(), loss_count.item()

    # ----- evaluation --------------------------------------------------------

    def evaluate(self, val_loader, criterion_cls) -> Dict[str, float]:
        if not val_loader:
            return {"val_loss": 0, "val_acc": 0, "val_mae": 0, "val_rmse": 0}

        self.model.eval()
        correct, total, loss_sum, mae_sum, mse_sum = 0, 0, 0.0, 0.0, 0.0
        autocast_enabled = torch.cuda.is_available()
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                density_target = batch["density_target"].to(self.device, non_blocking=True)
                count_target = batch["count"].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=autocast_enabled):
                    logits, density_pred = self.model(pixel_values)
                    total_loss, _, _, _ = self._multi_task_loss(
                        logits,
                        density_pred,
                        labels,
                        density_target,
                        count_target,
                        criterion_cls,
                    )
                loss_sum += total_loss.item() * labels.size(0)

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                count_pred = density_pred.flatten(1).sum(dim=1)
                diff = count_pred - count_target
                mae_sum += diff.abs().sum().item()
                mse_sum += (diff * diff).sum().item()

        return {
            "val_loss": loss_sum / max(total, 1),
            "val_acc": correct / max(total, 1),
            "val_mae": mae_sum / max(total, 1),
            "val_rmse": math.sqrt(mse_sum / max(total, 1)),
        }

    # ----- inference ---------------------------------------------------------

    def infer(self, image):
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            logits, density_pred = self.model(inputs["pixel_values"])
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf = probs[0, pred].item()
            pred_class = self.classes[pred]
            est_count = float(density_pred.flatten(1).sum(dim=1).item())
            return pred_class, conf, est_count

    # ----- checkpoint helper -------------------------------------------------

    def _save(self, path, optimizer, scheduler, classes, step, epoch):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {
                    "model_name": self.model_name,
                    "classes": classes,
                    "backbone": self.backbone_config,
                    "image_processor": self.image_processor_config,
                    "freeze_backbone": self.freeze_backbone,
                    "density_map_size": self.density_map_size,
                },
                "step": step,
                "epoch": epoch,
            },
            path,
        )

    # ----- training loop -----------------------------------------------------

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        lr,
        eval_every_steps,
        weight_decay,
        warmup_ratio,
        checkpoint_dir,
        classes,
        density_lr=None,
        density_weight_decay=None,
        backbone_lr=None,
    ):
        self.classes = classes
        # Per-head LR + WD: classification head uses `lr` / `weight_decay`,
        # density head uses `density_lr` / `density_weight_decay` (each falling
        # back to the shared value when None). When the backbone is unfrozen,
        # a third param group covers the backbone at `backbone_lr` (defaults to
        # density_lr). Param groups inherit the same cosine warmup multiplier
        # so all heads decay in sync.
        cls_params = [p for p in self.model.cls_head.parameters() if p.requires_grad]
        density_params = [p for p in self.model.density_head.parameters() if p.requires_grad]
        effective_density_lr = density_lr if density_lr is not None else lr
        effective_density_wd = density_weight_decay if density_weight_decay is not None else weight_decay
        param_groups = [
            {"params": cls_params, "lr": lr, "weight_decay": weight_decay},
            {"params": density_params, "lr": effective_density_lr, "weight_decay": effective_density_wd},
        ]
        backbone_lr_used = None
        if not self.freeze_backbone:
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
            backbone_lr_used = backbone_lr if backbone_lr is not None else effective_density_lr
            param_groups.append({"params": backbone_params, "lr": backbone_lr_used, "weight_decay": weight_decay})
        optimizer = torch.optim.AdamW(param_groups)
        if self.logging:
            msg = f"Optimizer: cls_head lr={lr} wd={weight_decay} | " f"density_head lr={effective_density_lr} wd={effective_density_wd}"
            if backbone_lr_used is not None:
                msg += f" | backbone lr={backbone_lr_used} wd={weight_decay} (UNFROZEN)"
            self.logging.info(msg)
        total_steps = epochs * math.ceil(len(train_loader))
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        class_weights_tensor = None
        if self.class_weights is not None:
            class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32)
        criterion_cls = LabelDistributionLoss(
            num_classes=self.num_classes,
            sigma=self.ldl_sigma,
            class_weights=class_weights_tensor,
        ).to(self.device)

        autocast_enabled = torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)

        best_acc = 0.0
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            if self.freeze_backbone:
                self.model.backbone.eval()  # keep frozen backbone in eval mode

            running_total = running_cls = running_density = running_count = 0.0

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                density_target = batch["density_target"].to(self.device, non_blocking=True)
                count_target = batch["count"].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=autocast_enabled):
                    logits, density_pred = self.model(pixel_values)
                    total_loss, l_cls, l_dens, l_cnt = self._multi_task_loss(
                        logits,
                        density_pred,
                        labels,
                        density_target,
                        count_target,
                        criterion_cls,
                    )

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_total += total_loss.item()
                running_cls += l_cls
                running_density += l_dens
                running_count += l_cnt
                global_step += 1

                if global_step % eval_every_steps == 0:
                    metrics = self.evaluate(val_loader, criterion_cls)
                    self.logging.info(f"[epoch {epoch} | step {global_step}] " f"train_total={running_total/eval_every_steps:.4f} " f"train_cls={running_cls/eval_every_steps:.4f} " f"train_dens={running_density/eval_every_steps:.4f} " f"train_cnt={running_count/eval_every_steps:.4f} " f"val_loss={metrics['val_loss']:.4f} " f"val_acc={metrics['val_acc']*100:.2f}% " f"val_count_mae={metrics['val_mae']:.2f} " f"val_count_rmse={metrics['val_rmse']:.2f}")
                    running_total = running_cls = running_density = running_count = 0.0

                    if metrics["val_acc"] >= best_acc:
                        best_acc = metrics["val_acc"]
                        self._save(
                            os.path.join(checkpoint_dir, self._ckpt_filename),
                            optimizer,
                            scheduler,
                            classes,
                            global_step,
                            epoch,
                        )

            metrics = self.evaluate(val_loader, criterion_cls)
            if metrics["val_acc"] >= best_acc:
                best_acc = metrics["val_acc"]
                self._save(
                    os.path.join(checkpoint_dir, self._ckpt_filename),
                    optimizer,
                    scheduler,
                    classes,
                    global_step,
                    epoch,
                )
            self.logging.info(f"END EPOCH {epoch}: val_loss={metrics['val_loss']:.4f} " f"val_acc={metrics['val_acc']*100:.2f}% " f"val_count_mae={metrics['val_mae']:.2f} " f"val_count_rmse={metrics['val_rmse']:.2f} " f"(best_acc={best_acc*100:.2f}%)")

        # Load the best-val checkpoint before the final test evaluation. The
        # final-epoch model state can drift below the peak val_acc (val_acc
        # oscillates while train_loss keeps falling), so we evaluate the
        # checkpoint we actually save and ship.
        best_ckpt_path = os.path.join(checkpoint_dir, self._ckpt_filename)
        if os.path.isfile(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if self.logging:
                self.logging.info(f"END TRAIN: loaded best-val checkpoint " f"(epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')}, best_val_acc={best_acc*100:.2f}%)")
        metrics = self.evaluate(test_loader, criterion_cls)
        self.logging.info(f"END TRAIN: test_loss={metrics['val_loss']:.4f} " f"test_acc={metrics['val_acc']*100:.2f}% " f"test_count_mae={metrics['val_mae']:.2f} " f"test_count_rmse={metrics['val_rmse']:.2f}")
