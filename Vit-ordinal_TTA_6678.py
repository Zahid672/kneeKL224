"""
ViT-B/16 + CORAL-style Ordinal Head for KL Grade (0..4)

Improvements:
- Stratified K-fold (no balancing) over train+val.
- ViT-B/16 with CORAL ordinal head (4 logits for y>0..3).
- Image size 224x224 (required by your torchvision ViT).
- Longer training (80 epochs) with early stopping.
- Class-weighted BCE for ordinal loss (handles class imbalance).
- Test-time augmentation (TTA) on test set (identity + flip + ±10°).
- Saves per-fold + test metrics to CSV.
- Saves test predictions to CSV.

Folder structure:

DATA_ROOT/
    train/
        0/, 1/, 2/, 3/, 4/
    val/
        0/, 1/, 2/, 3/, 4/
    test/
        0/, 1/, 2/, 3/, 4/
"""

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from torch.utils.tensorboard import SummaryWriter
from PIL import Image


# =========================
# CONFIGURATION
# =========================
DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"  

NUM_CLASSES = 5            # KL grades 0..4
NUM_FOLDS = 5
IMG_SIZE = 224             # vit_b_16 expects 224x224

BATCH_SIZE = 16
NUM_EPOCHS = 80            # longer, with early stopping
LR = 3e-5
WEIGHT_DECAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

EARLY_STOP_PATIENCE = 10   # epochs with no val improvement
FEATURE_EXTRACT = False    # if True, freeze ViT backbone

SEED = 42

METRICS_CSV = "ViT_B16_CORAL_kfold_results_weighted_TTA.csv"
PREDICTIONS_CSV = "ViT_B16_CORAL_bestFold_test_predictions_weighted_TTA.csv"
BEST_MODEL_PATH = "ViT_B16_CORAL_bestFold_weighted_TTA.pth"


# =========================
# UTILS
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def specificity_score(y_true, y_pred, num_classes, average="macro"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    TN, FP = [], []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))
        TN.append(tn)
        FP.append(fp)
    spec = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-8)
    if average == "macro":
        return float(np.mean(spec))
    return spec


def compute_pos_weight(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute per-threshold pos_weight for CORAL:
    For threshold k (y > k):
        pos = #samples with y > k
        neg = #samples with y <= k
    pos_weight_k = neg / pos
    """
    pos_weights = []
    total = len(labels)
    for k in range(num_classes - 1):
        pos = int(np.sum(labels > k))
        neg = total - pos
        if pos == 0:
            pos_weight_k = 1.0
        else:
            pos_weight_k = neg / float(pos)
        pos_weights.append(pos_weight_k)
    return torch.tensor(pos_weights, dtype=torch.float32)


# =========================
# DATASETS
# =========================
class PathLabelDataset(Dataset):
    """Dataset from (path, label) pairs + transform."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def load_trainval_samples_and_labels(root: str):
    """Read file paths + labels from train+val folders and merge."""
    train_folder = datasets.ImageFolder(os.path.join(root, "train"))
    val_folder = datasets.ImageFolder(os.path.join(root, "val"))

    train_samples = train_folder.samples
    val_samples = val_folder.samples
    samples = train_samples + val_samples

    train_targets = train_folder.targets
    val_targets = val_folder.targets
    labels = np.array(train_targets + val_targets, dtype=np.int64)

    return samples, labels


def load_test_dataset(root: str):
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(root, "test"),
                                   transform=test_tf)
    return test_ds


def make_loader(dataset: Dataset, indices, shuffle: bool):
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader


# =========================
# MODEL: ViT-B/16 + CORAL Ordinal Head
# =========================
class ViT_CORAL(nn.Module):
    """
    ViT-B/16 backbone with CORAL-style ordinal head.

    For K=5 classes, we predict K-1=4 logits: z1..z4 for events:
        y > 0, y > 1, y > 2, y > 3

    Training target for sample with grade y:
        t_k = 1 if y > k else 0, for k=0..3
    """

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        num_thresholds = num_classes - 1

        # ViT backbone
        try:
            vit = models.vit_b_16(pretrained=pretrained)
        except TypeError:
            vit = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )

        embed_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()          # remove default classifier

        self.backbone = vit
        self.classifier = nn.Linear(embed_dim, num_thresholds)

    def forward(self, x):
        feats = self.backbone(x)          # [B, embed_dim]
        logits = self.classifier(feats)   # [B, K-1]
        return logits

    @torch.no_grad()
    def predict_from_logits(self, logits):
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs >= 0.5, dim=1).long()
        return preds

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        return self.predict_from_logits(logits)


def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for p in model.backbone.parameters():
            p.requires_grad = False


# =========================
# TRAIN / EVAL
# =========================
def labels_to_coral_targets(labels: torch.Tensor, num_classes: int):
    """
    Convert integer labels in {0..K-1} to CORAL ordinal targets of shape [B, K-1].
    t_k = 1 if label > k else 0.
    """
    labels = labels.unsqueeze(1)  # [B, 1]
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    targets = (labels > thresholds).float()  # [B, K-1]
    return targets


def train_one_epoch(
    model,
    loader,
    optimizer,
    bce_loss,
    device,
    num_classes,
    writer=None,
    epoch_idx=None,
    fold_idx=None,
):
    model.train()
    running = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        logits = model(images)                         # [B, K-1]
        targets = labels_to_coral_targets(labels, num_classes)  # [B, K-1]

        loss = bce_loss(logits, targets)
        loss.backward()
        optimizer.step()

        running += loss.item() * labels.size(0)

    epoch_loss = running / len(loader.dataset)
    if writer is not None and epoch_idx is not None:
        writer.add_scalar(f"Fold{fold_idx}/Loss_train", epoch_loss, epoch_idx)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    pos_weight,
    writer=None,
    epoch_idx=None,
    fold_idx=None,
    split_name="val",
):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    total = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        targets = labels_to_coral_targets(labels, num_classes)
        loss = bce_loss(logits, targets)
        total += loss.item()

        preds = model.predict_from_logits(logits)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))

    if writer is not None and epoch_idx is not None:
        writer.add_scalar(f"Fold{fold_idx}/Loss_{split_name}", avg_loss, epoch_idx)
        writer.add_scalar(f"Fold{fold_idx}/Acc_{split_name}", acc, epoch_idx)

    return avg_loss, acc, y_true, y_pred


@torch.no_grad()
def evaluate_tta(
    model,
    loader,
    device,
    num_classes,
    pos_weight,
    n_augs: int = 4,
):
    """
    Test-time augmentation evaluation.

    Augs used:
        - identity
        - horizontal flip
        - rotate +10
        - rotate -10
    """
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    total = 0.0
    all_labels, all_preds = [], []

    def apply_augs(x):
        # x: [B, C, H, W]
        augs = []
        augs.append(x)                      # identity
        augs.append(F.hflip(x))            # horizontal flip
        augs.append(F.rotate(x, 10))       # +10 deg
        augs.append(F.rotate(x, -10))      # -10 deg
        return augs[:n_augs]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits_sum = None
        for aug_imgs in apply_augs(images):
            logits_aug = model(aug_imgs)
            if logits_sum is None:
                logits_sum = logits_aug
            else:
                logits_sum = logits_sum + logits_aug
        logits_mean = logits_sum / float(n_augs)

        targets = labels_to_coral_targets(labels, num_classes)
        loss = bce_loss(logits_mean, targets)
        total += loss.item()

        preds = model.predict_from_logits(logits_mean)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))

    return avg_loss, acc, y_true, y_pred


def compute_metrics(y_true, y_pred, num_classes):
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "specificity_macro": specificity_score(y_true, y_pred, num_classes),
    }

    y_true_oh = np.eye(num_classes)[y_true]
    y_pred_oh = np.eye(num_classes)[y_pred]
    try:
        metrics["auroc_macro"] = float(
            roc_auc_score(y_true_oh, y_pred_oh, average="macro")
        )
    except Exception:
        metrics["auroc_macro"] = 0.0

    return metrics


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    print("Using device:", DEVICE)

    # 1) Load samples/labels + test dataset
    samples, labels = load_trainval_samples_and_labels(DATA_ROOT)
    labels = np.asarray(labels, dtype=np.int64)

    test_ds = load_test_dataset(DATA_ROOT)

    # Class weights for CORAL thresholds
    pos_weight = compute_pos_weight(labels, NUM_CLASSES).to(DEVICE)
    print("pos_weight for thresholds (y>0..3):", pos_weight.tolist())

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PathLabelDataset(samples, transform=train_tf)
    val_ds = PathLabelDataset(samples, transform=val_tf)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # 2) Stratified K-Fold
    indices = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_metrics_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(indices, labels), start=1
    ):
        print(f"\n========== FOLD {fold_idx}/{NUM_FOLDS} ==========")

        uniq_tr, cnt_tr = np.unique(labels[train_idx], return_counts=True)
        print("Train class counts:", dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))
        uniq_val, cnt_val = np.unique(labels[val_idx], return_counts=True)
        print("Val class counts:", dict(zip(uniq_val.tolist(), cnt_val.tolist())))

        train_loader = make_loader(train_ds, train_idx, shuffle=True)
        val_loader = make_loader(val_ds, val_idx, shuffle=False)

        model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        set_parameter_requires_grad(model, FEATURE_EXTRACT)

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS
        )

        bce_loss_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

        writer = SummaryWriter(
            log_dir=f"runs/ViT_B16_CORAL_weighted/fold_{fold_idx}"
        )

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Fold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, bce_loss_train,
                DEVICE, NUM_CLASSES, writer, epoch, fold_idx
            )
            val_loss, val_acc, y_val, y_val_pred = evaluate(
                model, val_loader, DEVICE, NUM_CLASSES, pos_weight,
                writer, epoch, fold_idx, split_name="val"
            )

            print(f"  train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "fold": fold_idx,
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping on fold {fold_idx} at epoch {epoch}")
                break

        writer.close()

        if best_state is None:
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "fold": fold_idx,
            }

        model.load_state_dict(best_state["model"])
        val_loss, val_acc, y_val, y_val_pred = evaluate(
            model, val_loader, DEVICE, NUM_CLASSES, pos_weight
        )

        fold_metrics = compute_metrics(y_val, y_val_pred, NUM_CLASSES)
        fold_metrics["fold"] = fold_idx
        fold_metrics["split"] = "val"
        fold_metrics["val_loss"] = float(val_loss)
        all_metrics_rows.append(fold_metrics)

        print("\nFold", fold_idx, "validation confusion matrix:")
        print(confusion_matrix(y_val, y_val_pred))
        print("\nFold", fold_idx, "validation classification report:")
        print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_overall_state = best_state
            best_fold_idx = fold_idx

    print(f"\nBest fold: {best_fold_idx} with val_acc={best_overall_val_acc:.4f}")
    torch.save(best_overall_state, BEST_MODEL_PATH)
    print(f"Best fold model saved to {BEST_MODEL_PATH}")

    # 3) Final test with best fold model + TTA
    best_model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(best_overall_state["model"])

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    test_loss, test_acc, y_test, y_test_pred = evaluate_tta(
        best_model, test_loader, DEVICE, NUM_CLASSES, pos_weight, n_augs=4
    )
    test_metrics = compute_metrics(y_test, y_test_pred, NUM_CLASSES)
    test_metrics["fold"] = best_fold_idx
    test_metrics["split"] = "test_TTA"
    test_metrics["val_loss"] = float(test_loss)
    all_metrics_rows.append(test_metrics)

    print("\n===== TEST RESULTS (best fold model, with TTA) =====")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    # 4) Save metrics & predictions
    df_metrics = pd.DataFrame(all_metrics_rows)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nAll fold + test metrics saved to {METRICS_CSV}")

    df_pred = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_test_pred,
    })
    df_pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Test predictions saved to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
