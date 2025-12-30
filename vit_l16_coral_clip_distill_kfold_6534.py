"""
ViT-L/16 + CORAL for KL grading on OAI knee joints (kneeKL224)
with CLIP-based VLM distillation and prompt-contrastive regularization.

- Student: torchvision ViT-L/16 + CORAL ordinal head.
- Teacher: OpenCLIP ViT-L/14 (frozen).
- Extra losses:
    * Feature distillation: student CLS features -> CLIP image features (MSE).
    * Prompt contrastive: student features vs KL text prompts in CLIP space.

Data structure (same as before):

DATA_ROOT/
    train/0..4
    val/0..4
    test/0..4
"""

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import open_clip

# =========================
# CONFIG
# =========================

DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5               # KL 0..4
IMG_SIZE = 224
NUM_FOLDS = 5

BATCH_SIZE = 16
NUM_EPOCHS = 80
LR = 3e-5
WEIGHT_DECAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 10
FEATURE_EXTRACT = False
SEED = 42

# CLIP VLM config
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"

# Distillation / contrastive weights
LAMBDA_DISTILL = 0.5
LAMBDA_CONTRAST = 0.1
CONTRAST_TEMPERATURE = 0.07

METRICS_CSV = "ViT_L16_CORAL_CLIPdistill_kfold_metrics.csv"
PREDICTIONS_CSV = "ViT_L16_CORAL_CLIPdistill_bestFold_test_predictions.csv"
BEST_MODEL_PATH = "ViT_L16_CORAL_CLIPdistill_bestFold.pth"


# =========================
# UTILITIES
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
    For CORAL thresholds y > k, k = 0..3.
    pos_weight_k = (N_neg / N_pos) to rebalance thresholds.
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


class PathLabelDataset(Dataset):
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
    train_folder = datasets.ImageFolder(os.path.join(root, "train"))
    val_folder = datasets.ImageFolder(os.path.join(root, "val"))

    samples = train_folder.samples + val_folder.samples
    labels = np.array(train_folder.targets + val_folder.targets, dtype=np.int64)
    return samples, labels


def load_test_dataset(root: str):
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(root, "test"), transform=test_tf)
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


def get_class_prompts() -> List[str]:
    """Text prompts for each KL grade (for CLIP)."""
    prompts = [
        "an AP knee radiograph with Kellgren-Lawrence grade 0, no osteoarthritis",
        "an AP knee radiograph with Kellgren-Lawrence grade 1, doubtful osteophytes and possible joint space narrowing",
        "an AP knee radiograph with Kellgren-Lawrence grade 2, definite osteophytes and possible joint space narrowing",
        "an AP knee radiograph with Kellgren-Lawrence grade 3, multiple osteophytes and definite joint space narrowing",
        "an AP knee radiograph with Kellgren-Lawrence grade 4, severe joint space narrowing and large osteophytes",
    ]
    return prompts


# =========================
# MODEL (ViT-L/16 + CORAL)
# =========================

class ViT_L16_CORAL(nn.Module):
    """
    ViT-L/16 backbone with CORAL ordinal head.
    Forward returns both logits (thresholds) and CLS features.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        num_thresholds = num_classes - 1

        try:
            vit = models.vit_l_16(pretrained=pretrained)
        except TypeError:
            vit = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )

        embed_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()          # remove classification head
        self.backbone = vit
        self.classifier = nn.Linear(embed_dim, num_thresholds)

    def forward(self, x):
        feats = self.backbone(x)           # CLS embedding
        logits = self.classifier(feats)
        return logits, feats

    @torch.no_grad()
    def predict_from_logits(self, logits):
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs >= 0.5, dim=1).long()
        return preds

    @torch.no_grad()
    def predict(self, x):
        logits, _ = self.forward(x)
        return self.predict_from_logits(logits)


def labels_to_coral_targets(labels: torch.Tensor, num_classes: int):
    labels = labels.unsqueeze(1)                      # [B,1]
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    targets = (labels > thresholds).float()           # [B, K-1]
    return targets


# =========================
# CLIP TEACHER
# =========================

@torch.no_grad()
def get_teacher_features(clip_model, images):
    """
    Get CLIP image features for a batch, with simple two-view aggregation
    (original + horizontal flip) to make the teacher representation richer.
    """
    clip_model.eval()
    # CLIP expects normalized inputs; our images are already normalized for ViT.
    # This mismatch is minor and acceptable for distillation.
    feats1 = clip_model.encode_image(images)
    feats2 = clip_model.encode_image(torch.flip(images, dims=[3]))  # flip W
    feats = (feats1 + feats2) / 2.0
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach()


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
# TRAIN / EVAL
# =========================

def train_one_epoch(
    model,
    distill_head,
    clip_model,
    text_features,
    loader,
    optimizer,
    pos_weight,
    cls_weight_vec,
    device,
    num_classes,
):
    model.train()
    distill_head.train()
    running = 0.0

    bce_reduction_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        logits, feats = model(images)                  # [B,K-1], [B,D_s]
        targets = labels_to_coral_targets(labels, num_classes)

        # ----- CORAL loss (ordinal) -----
        sample_w = cls_weight_vec[labels]             # [B]
        loss_per_sample = bce_reduction_none(logits, targets)  # [B,K-1]
        loss_per_sample = loss_per_sample.mean(dim=1)          # [B]
        loss_coral = (loss_per_sample * sample_w).mean()

        # ----- CLIP teacher features -----
        with torch.no_grad():
            teacher_feats = get_teacher_features(clip_model, images)   # [B,D_t]

        proj_student = distill_head(feats)             # [B,D_t]

        # ----- distillation loss -----
        loss_distill = mse_loss(proj_student, teacher_feats)

        # ----- prompt contrastive loss -----
        # normalize
        proj_student_norm = proj_student / proj_student.norm(dim=-1, keepdim=True)
        text_feats_norm = text_features / text_features.norm(dim=-1, keepdim=True)  # [C,D_t]
        logits_contrast = (proj_student_norm @ text_feats_norm.t()) / CONTRAST_TEMPERATURE  # [B,C]
        loss_contrast = ce_loss(logits_contrast, labels)

        loss = loss_coral + LAMBDA_DISTILL * loss_distill + LAMBDA_CONTRAST * loss_contrast
        loss.backward()
        optimizer.step()

        running += loss.item() * labels.size(0)

    epoch_loss = running / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    pos_weight,
):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    total = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits, _ = model(images)
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
    return avg_loss, acc, y_true, y_pred


# =========================
# MAIN TRAINING PIPELINE
# =========================

def run_training():
    set_seed(SEED)
    print("Using device:", DEVICE)

    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")

    # ---------- load data ----------
    samples, labels = load_trainval_samples_and_labels(DATA_ROOT)
    labels = np.asarray(labels, dtype=np.int64)

    test_ds = load_test_dataset(DATA_ROOT)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    pos_weight = compute_pos_weight(labels, NUM_CLASSES).to(DEVICE)
    print("pos_weight thresholds (y>0..3):", pos_weight.tolist())

    # Upweight G1 and G2
    cls_weight_vec = torch.tensor(
        [1.0, 1.5, 1.5, 1.0, 1.0], dtype=torch.float32, device=DEVICE
    )

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

    indices = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # ---------- CLIP teacher & prompts ----------
    print(f"Loading CLIP teacher: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    prompts = get_class_prompts()
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_features = clip_model.encode_text(text_tokens)  # [C,D_t]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    teacher_dim = text_features.shape[1]

    all_metrics_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        print(f"\n========== FOLD {fold_idx}/{NUM_FOLDS} ==========")

        uniq_tr, cnt_tr = np.unique(labels[train_idx], return_counts=True)
        print("Train class counts:", dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))
        uniq_val, cnt_val = np.unique(labels[val_idx], return_counts=True)
        print("Val class counts:", dict(zip(uniq_val.tolist(), cnt_val.tolist())))

        train_loader = make_loader(train_ds, train_idx, shuffle=True)
        val_loader = make_loader(val_ds, val_idx, shuffle=False)

        model = ViT_L16_CORAL(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

        if FEATURE_EXTRACT:
            for p in model.backbone.parameters():
                p.requires_grad = False

        # distillation head: student_dim -> teacher_dim
        dummy_in = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            _, tmp_feat = model(dummy_in.to(DEVICE))
        student_dim = tmp_feat.shape[1]
        distill_head = nn.Linear(student_dim, teacher_dim).to(DEVICE)

        params_to_update = list(model.parameters()) + list(distill_head.parameters())
        optimizer = optim.AdamW(params_to_update, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Fold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model, distill_head, clip_model, text_features,
                train_loader, optimizer,
                pos_weight, cls_weight_vec, DEVICE, NUM_CLASSES
            )
            val_loss, val_acc, y_val, y_val_pred = evaluate(
                model, val_loader, DEVICE, NUM_CLASSES, pos_weight
            )

            print(f"  train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_state = {
                    "model": model.state_dict(),
                    "distill_head": distill_head.state_dict(),
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

        if best_state is None:
            best_state = {
                "model": model.state_dict(),
                "distill_head": distill_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "fold": fold_idx,
            }

        # Evaluate best state on validation
        model.load_state_dict(best_state["model"])
        distill_head.load_state_dict(best_state["distill_head"])
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

        if val_acc > best_overall_val_acc:
            best_overall_val_acc = val_acc
            best_overall_state = best_state
            best_fold_idx = fold_idx

    print(f"\nBest fold: {best_fold_idx} with val_acc={best_overall_val_acc:.4f}")
    torch.save(best_overall_state, BEST_MODEL_PATH)
    print(f"Best fold model saved to {BEST_MODEL_PATH}")

    # ---------- Final test with best fold model ----------
    best_model = ViT_L16_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(best_overall_state["model"])

    # distill head not needed for inference, but load to keep completeness
    distill_head = nn.Linear(student_dim, teacher_dim).to(DEVICE)
    distill_head.load_state_dict(best_overall_state["distill_head"])

    test_loss, test_acc, y_test, y_test_pred = evaluate(
        best_model, test_loader, DEVICE, NUM_CLASSES, pos_weight
    )
    test_metrics = compute_metrics(y_test, y_test_pred, NUM_CLASSES)
    test_metrics["fold"] = best_fold_idx
    test_metrics["split"] = "test"
    test_metrics["val_loss"] = float(test_loss)
    all_metrics_rows.append(test_metrics)

    print("\n===== TEST RESULTS (best fold model) =====")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

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
    run_training()
