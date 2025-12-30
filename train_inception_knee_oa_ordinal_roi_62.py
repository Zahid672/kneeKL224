"""
Ordinal Inception V3 with automatic knee-joint ROI cropping
for KL grade classification (0..4).

- On-the-fly knee joint ROI detection from full X-ray
- CLAHE on ROI
- Inception V3 with ordinal head (4 logits for 5 ordered classes)
- WeightedRandomSampler for class balancing
- Prints confusion matrix + per-class + global metrics

Expected directory structure:

DATA_ROOT/
    train/
        0/ 1/ 2/ 3/ 4/
    val/
        0/ 1/ 2/ 3/ 4/
    test/
        0/ 1/ 2/ 3/ 4/
"""

import os
import time
from copy import deepcopy

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_ROOT = r"E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5                  # KL grades 0..4
NUM_THRESHOLDS = NUM_CLASSES - 1 # 4 logits for ordinal head
BATCH_SIZE = 32
EPOCHS = 70

# MODE:
#   "IV3"    - feature extraction (only head learns)
#   "FT_IV3" - full fine-tuning (recommended)
MODE = "FT_IV3"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------
# KNEE-JOINT ROI + CLAHE PREPROCESSING
# ---------------------------------------------------------------------

def knee_joint_roi_preprocess(pil_img: Image.Image,
                              band_height_ratio: float = 0.6) -> Image.Image:
    """
    1) Convert to grayscale
    2) Use Sobel-y to find the horizontal band with strongest edges (joint space)
    3) Crop a vertical strip centered on that band
    4) Apply CLAHE
    5) Return 3-channel RGB PIL image
    """
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    # Sobel in y to emphasize horizontal edges (joint gap)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    edge_mag = np.abs(sobel_y)

    row_strength = edge_mag.mean(axis=1)
    row_strength_smooth = cv2.GaussianBlur(row_strength[:, None], (15, 1), 0).squeeze()

    center_row = int(np.argmax(row_strength_smooth))

    half_band = int(h * band_height_ratio / 2.0)
    y1 = max(center_row - half_band, 0)
    y2 = min(center_row + half_band, h)

    # Fallback if something went weird
    if y2 <= y1 + 10:
        y1 = int(0.2 * h)
        y2 = int(0.8 * h)

    crop = gray[y1:y2, :]

    # CLAHE on cropped ROI
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(crop)

    eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(eq_rgb)

# ---------------------------------------------------------------------
# DATASETS AND DATALOADERS
# ---------------------------------------------------------------------

def get_dataloaders():
    input_size = 299  # Inception V3 default

    train_transform = transforms.Compose([
        transforms.Lambda(knee_joint_roi_preprocess),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(knee_joint_roi_preprocess),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir   = os.path.join(DATA_ROOT, "val")
    test_dir  = os.path.join(DATA_ROOT, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    # Balanced sampling for train
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    print("Train class counts:", class_counts.tolist())

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    print("Dataset sizes:")
    print("  train:", dataset_sizes["train"])
    print("  val:  ", dataset_sizes["val"])
    print("  test: ", len(test_dataset))
    print("  classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader, dataset_sizes

# ---------------------------------------------------------------------
# ORDINAL HELPERS
# ---------------------------------------------------------------------

_bce = nn.BCEWithLogitsLoss()

def labels_to_ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer labels 0..K-1 to K-1 binary targets:

    For K=5 (0..4):
      y=0 -> [0,0,0,0]
      y=1 -> [1,0,0,0]
      y=2 -> [1,1,0,0]
      y=3 -> [1,1,1,0]
      y=4 -> [1,1,1,1]
    """
    K = num_classes
    labels = labels.unsqueeze(1)  # [N,1]
    thresholds = torch.arange(K-1, device=labels.device).unsqueeze(0)  # [1,K-1]
    targets = (labels > thresholds).float()  # [N,K-1]
    return targets

def ordinal_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = labels_to_ordinal_targets(labels, num_classes)
    return _bce(logits, targets)

def ordinal_predict_classes(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)           # [N,K-1]
    preds = torch.sum(probs > 0.5, dim=1)   # 0..K-1
    return preds

def ordinal_probabilities(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert p_gt[k] = P(y > k) into per-class probabilities.

    For K=5, p0..p3:
      P(y=0) = 1 - p0
      P(y=1) = p0 - p1
      P(y=2) = p1 - p2
      P(y=3) = p2 - p3
      P(y=4) = p3
    """
    K = num_classes
    p_gt = torch.sigmoid(logits)  # [N,K-1]
    N = p_gt.size(0)
    p = torch.zeros(N, K, device=logits.device)

    p[:, 0] = 1.0 - p_gt[:, 0]
    for k in range(1, K-1):
        p[:, k] = p_gt[:, k-1] - p_gt[:, k]
    p[:, K-1] = p_gt[:, K-2]

    p = torch.clamp(p, 1e-6, 1.0)
    p = p / p.sum(dim=1, keepdim=True)
    return p

# ---------------------------------------------------------------------
# MODEL: INCEPTION V3 + ORDINAL HEAD
# ---------------------------------------------------------------------

def build_inception_v3_ordinal(num_thresholds: int, mode: str = "IV3"):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, aux_logits=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_thresholds)  # 4 logits

    if mode == "IV3":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif mode == "FT_IV3":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("mode must be 'IV3' or 'FT_IV3'")

    return model

# ---------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------

from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(model, dataloaders, dataset_sizes,
                optimizer, scheduler,
                num_epochs=25, device="cpu"):

    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*32)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # handle aux_logits: use main output only
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    loss = ordinal_loss(logits, labels, NUM_CLASSES)
                    preds = ordinal_predict_classes(logits)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # track best val accuracy
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

        scheduler.step()

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed/60:.1f} minutes")
    print(f"Best val Acc: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# ---------------------------------------------------------------------
# TEST EVALUATION
# ---------------------------------------------------------------------

def evaluate_on_test(model, test_loader, device="cpu"):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            preds = ordinal_predict_classes(logits)
            probs = ordinal_probabilities(logits, NUM_CLASSES)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_score = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print("\nConfusion Matrix:")
    print(cm)

    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    print("\nPer-Class Metrics:")
    for i in range(NUM_CLASSES):
        print(
            f"Class {i}: Precision={per_class_prec[i]:.4f}, "
            f"Recall={per_class_rec[i]:.4f}, F1={per_class_f1[i]:.4f}"
        )

    # specificity
    spec_per_class = []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        denom = TN + FP
        spec = TN / denom if denom > 0 else 0.0
        spec_per_class.append(spec)
    specificity = float(np.mean(spec_per_class))

    try:
        auroc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    print("\n===== TEST METRICS =====")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {precision:.4f} (macro)")
    print(f"Sensitivity:  {recall:.4f} (macro recall)")
    print(f"F1-score:     {f1:.4f} (macro)")
    print(f"Specificity:  {specificity:.4f} (macro)")
    print(f"AUROC (OVR):  {auroc:.4f}")
    print("========================\n")

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "specificity_macro": specificity,
        "auroc_macro_ovr": auroc,
        "confusion_matrix": cm,
    }

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset_sizes = get_dataloaders()
    dataloaders = {"train": train_loader, "val": val_loader}

    model = build_inception_v3_ordinal(NUM_THRESHOLDS, mode=MODE).to(device)

    lr = 5e-4 if MODE == "IV3" else 1e-4
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nMode: {MODE}, lr: {lr}, trainable params: {len(params_to_update)}")

    model = train_model(
        model,
        dataloaders,
        dataset_sizes,
        optimizer,
        scheduler,
        num_epochs=EPOCHS,
        device=device,
    )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"inceptionv3_ORD_ROI_{MODE}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved best model to:", ckpt_path)

    _ = evaluate_on_test(model, test_loader, device=device)
