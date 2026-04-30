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
        density_map_size: int = 56,
        density_loss_weight: float = 100.0,
        count_loss_weight: float = 0.05,
    ):
        self.model_name = model_name
        self.logging = logging
        self.num_classes = num_classes
        self.classes = classes
        self.density_map_size = density_map_size
        self.density_loss_weight = density_loss_weight
        self.count_loss_weight = count_loss_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._ckpt_filename = model_name.replace("/", "_") + "_multitask_best.pt"

        if ckpt_path is None and checkpoint_dir is not None:
            candidate = os.path.join(checkpoint_dir, self._ckpt_filename)
            if os.path.isfile(candidate):
                ckpt_path = candidate
                if logging:
                    logging.info(f"Found existing checkpoint: {candidate}")

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
            self.freeze_backbone = True
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
            return {"val_loss": 0, "val_acc": 0, "val_mae": 0}

        self.model.eval()
        correct, total, loss_sum, mae_sum = 0, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                density_target = batch["density_target"].to(self.device, non_blocking=True)
                count_target = batch["count"].to(self.device, non_blocking=True)

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
                mae_sum += (count_pred - count_target).abs().sum().item()

        return {
            "val_loss": loss_sum / max(total, 1),
            "val_acc": correct / max(total, 1),
            "val_mae": mae_sum / max(total, 1),
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
    ):
        self.classes = classes
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        total_steps = epochs * math.ceil(len(train_loader))
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        criterion_cls = nn.CrossEntropyLoss()

        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        best_acc = 0.0
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.model.backbone.eval()  # keep frozen backbone in eval mode

            running_total = running_cls = running_density = running_count = 0.0

            for i, batch in enumerate(train_loader, start=1):
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                density_target = batch["density_target"].to(self.device, non_blocking=True)
                count_target = batch["count"].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
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
                    self.logging.info(
                        f"[epoch {epoch} | step {global_step}] "
                        f"train_total={running_total/eval_every_steps:.4f} "
                        f"train_cls={running_cls/eval_every_steps:.4f} "
                        f"train_dens={running_density/eval_every_steps:.4f} "
                        f"train_cnt={running_count/eval_every_steps:.4f} "
                        f"val_loss={metrics['val_loss']:.4f} "
                        f"val_acc={metrics['val_acc']*100:.2f}% "
                        f"val_count_mae={metrics['val_mae']:.2f}"
                    )
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
            self.logging.info(
                f"END EPOCH {epoch}: val_loss={metrics['val_loss']:.4f} " f"val_acc={metrics['val_acc']*100:.2f}% " f"val_count_mae={metrics['val_mae']:.2f} " f"(best_acc={best_acc*100:.2f}%)"
            )

        metrics = self.evaluate(test_loader, criterion_cls)
        self.logging.info(f"END TRAIN: test_loss={metrics['val_loss']:.4f} " f"test_acc={metrics['val_acc']*100:.2f}% " f"test_count_mae={metrics['val_mae']:.2f}")
