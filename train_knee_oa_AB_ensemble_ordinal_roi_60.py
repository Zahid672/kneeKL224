"""
Knee OA KL-grade (0..4) classification with:

A) Ensemble + Loss Reweighting
   - InceptionV3, EfficientNet-B3, ResNet50
   - Ordinal head (4 logits) for 5 ordered classes
   - Class-weighted ordinal loss (KL1 & KL2 up-weighted)

B) Class-specific augmentation
   - Stronger augmentation for KL1 and KL2
   - Moderate augmentation for KL0, KL3, KL4

Common:
   - Knee joint ROI cropping + CLAHE
   - WeightedRandomSampler for training
   - CSV saving for metrics and confusion matrix

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
import csv

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import (
    Inception_V3_Weights,
    EfficientNet_B3_Weights,
    ResNet50_Weights,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_ROOT = r"E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5
NUM_THRESHOLDS = NUM_CLASSES - 1  # 4 ordinal logits
BATCH_SIZE = 32
EPOCHS = 70

# "HEAD"  => only last layer trainable
# "FT"    => full fine-tuning (recommended)
MODE = "FT"

BACKBONES = ["inceptionv3", "efficientnet_b3", "resnet50"]

RESULTS_DIR = "results_AB"
CHECKPOINT_DIR = "checkpoints_AB"

# Class weights for loss (A: loss re-weighting)
# You can tune these; here KL1 & KL2 are emphasized.
CLASS_WEIGHTS_LIST = [1.0, 2.0, 2.0, 1.0, 1.2]  # [w0, w1, w2, w3, w4]

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------
# KNEE-JOINT ROI + CLAHE
# ---------------------------------------------------------------------

def knee_joint_roi_preprocess(pil_img: Image.Image,
                              band_height_ratio: float = 0.6) -> Image.Image:
    """
    1) Convert to grayscale
    2) Detect horizontal band with strongest edges (joint space)
    3) Crop a vertical strip around that band
    4) Apply CLAHE
    5) Return 3-channel RGB PIL image
    """
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    edge_mag = np.abs(sobel_y)

    row_strength = edge_mag.mean(axis=1)
    row_strength_smooth = cv2.GaussianBlur(row_strength[:, None], (15, 1), 0).squeeze()

    center_row = int(np.argmax(row_strength_smooth))

    half_band = int(h * band_height_ratio / 2.0)
    y1 = max(center_row - half_band, 0)
    y2 = min(center_row + half_band, h)

    # fallback if weird
    if y2 <= y1 + 10:
        y1 = int(0.2 * h)
        y2 = int(0.8 * h)

    crop = gray[y1:y2, :]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(crop)

    eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(eq_rgb)

# ---------------------------------------------------------------------
# CUSTOM DATASET: CLASS-SPECIFIC AUGMENTATION (B)
# ---------------------------------------------------------------------

class KneeKLDataset(Dataset):
    """
    Wraps ImageFolder for training.
    - Applies ROI + CLAHE
    - Uses stronger augmentation for labels 1 and 2.
    """
    def __init__(self, root, transform_common, transform_12):
        self.base = datasets.ImageFolder(root=root, transform=None)
        self.transform_common = transform_common
        self.transform_12 = transform_12
        # keep targets so WeightedRandomSampler can use them
        self.targets = [s[1] for s in self.base.samples]

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")

        roi = knee_joint_roi_preprocess(img)

        if label in (1, 2):
            img_t = self.transform_12(roi)
        else:
            img_t = self.transform_common(roi)

        return img_t, label

# ---------------------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------------------

def get_dataloaders():
    input_size = 299

    # moderate augmentation for classes 0,3,4
    transform_common = transforms.Compose([
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

    # stronger augmentation for classes 1,2
    transform_12 = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomRotation(degrees=25),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
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

    train_dataset = KneeKLDataset(train_dir, transform_common, transform_12)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    print("Train class counts:", class_counts.tolist())

    class_weights_sampler = 1.0 / class_counts
    sample_weights = class_weights_sampler[targets]
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
    print("  classes:", train_dataset.base.classes)

    return train_loader, val_loader, test_loader, dataset_sizes

# ---------------------------------------------------------------------
# ORDINAL HELPERS (with loss reweighting A)
# ---------------------------------------------------------------------

def labels_to_ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    K = num_classes
    labels = labels.unsqueeze(1)  # [N,1]
    thresholds = torch.arange(K-1, device=labels.device).unsqueeze(0)  # [1,K-1]
    targets = (labels > thresholds).float()  # [N,K-1]
    return targets

def ordinal_loss(logits: torch.Tensor,
                 labels: torch.Tensor,
                 num_classes: int,
                 class_weights: torch.Tensor = None) -> torch.Tensor:
    """
    BCE over thresholds, averaged per sample, with optional class weights.
    """
    targets = labels_to_ordinal_targets(labels, num_classes)  # [N, K-1]
    bce = nn.BCEWithLogitsLoss(reduction="none")
    loss_per_entry = bce(logits, targets)         # [N, K-1]
    loss_per_sample = loss_per_entry.mean(dim=1)  # [N]

    if class_weights is not None:
        weights = class_weights[labels]           # [N]
        loss_per_sample = loss_per_sample * weights

    return loss_per_sample.mean()

def ordinal_predict_classes(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)            # [N,K-1]
    preds = torch.sum(probs > 0.5, dim=1)    # 0..K-1
    return preds

def ordinal_probabilities(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
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
# MODEL BUILDERS
# ---------------------------------------------------------------------

def build_model(backbone: str, num_thresholds: int, mode: str = "FT") -> nn.Module:
    if backbone == "inceptionv3":
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_thresholds)

    elif backbone == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_thresholds)

    elif backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_thresholds)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if mode == "HEAD":
        for p in model.parameters():
            p.requires_grad = False
        # enable last head
        if backbone == "efficientnet_b3":
            for p in model.classifier[-1].parameters():
                p.requires_grad = True
        else:
            for p in model.fc.parameters():
                p.requires_grad = True
    else:  # full fine-tune
        for p in model.parameters():
            p.requires_grad = True

    return model

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_model(model, dataloaders, dataset_sizes,
                optimizer, scheduler,
                class_weights_tensor,
                num_epochs=25,
                device="cpu",
                backbone_name="model"):

    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n[{backbone_name}] Epoch {epoch+1}/{num_epochs}")
        print("-"*40)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]  # inception main output
                    else:
                        logits = outputs

                    loss = ordinal_loss(
                        logits,
                        labels,
                        NUM_CLASSES,
                        class_weights=class_weights_tensor,
                    )
                    preds = ordinal_predict_classes(logits)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

        scheduler.step()

    time_elapsed = time.time() - since
    print(f"\n[{backbone_name}] Training complete in {time_elapsed/60:.1f} minutes")
    print(f"[{backbone_name}] Best val Acc: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# ---------------------------------------------------------------------
# METRICS + CSV SAVING
# ---------------------------------------------------------------------

def compute_and_save_metrics(y_true, y_pred, y_score, tag: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    spec_per_class = []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        denom = TN + FP
        spec = TN / denom if denom > 0 else 0.0
        spec_per_class.append(spec)
    specificity_macro = float(np.mean(spec_per_class))

    try:
        auroc_macro = roc_auc_score(
            y_true, y_score, multi_class="ovr", average="macro"
        )
    except ValueError:
        auroc_macro = float("nan")

    print(f"\n[{tag}] Confusion Matrix:")
    print(cm)

    print(f"\n[{tag}] Per-Class Metrics:")
    for i in range(NUM_CLASSES):
        print(
            f"Class {i}: "
            f"P={per_class_prec[i]:.4f}, "
            f"R={per_class_rec[i]:.4f}, "
            f"F1={per_class_f1[i]:.4f}, "
            f"Spec={spec_per_class[i]:.4f}"
        )

    print(f"\n===== {tag} TEST METRICS =====")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {precision_macro:.4f} (macro)")
    print(f"Sensitivity:  {recall_macro:.4f} (macro recall)")
    print(f"F1-score:     {f1_macro:.4f} (macro)")
    print(f"Specificity:  {specificity_macro:.4f} (macro)")
    print(f"AUROC (OVR):  {auroc_macro:.4f}")
    print("================================\n")

    # overall metrics CSV
    overall_path = os.path.join(RESULTS_DIR, f"{tag}_overall_metrics.csv")
    with open(overall_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", acc])
        writer.writerow(["precision_macro", precision_macro])
        writer.writerow(["recall_macro", recall_macro])
        writer.writerow(["f1_macro", f1_macro])
        writer.writerow(["specificity_macro", specificity_macro])
        writer.writerow(["auroc_macro_ovr", auroc_macro])

    # per-class metrics CSV
    per_class_path = os.path.join(RESULTS_DIR, f"{tag}_per_class_metrics.csv")
    with open(per_class_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "specificity"])
        for i in range(NUM_CLASSES):
            writer.writerow([
                i,
                float(per_class_prec[i]),
                float(per_class_rec[i]),
                float(per_class_f1[i]),
                float(spec_per_class[i]),
            ])

    # confusion matrix CSV
    cm_path = os.path.join(RESULTS_DIR, f"{tag}_confusion_matrix.csv")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "specificity_macro": specificity_macro,
        "auroc_macro_ovr": auroc_macro,
        "confusion_matrix": cm,
    }

# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_single_model(model, test_loader, tag: str):
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

    return compute_and_save_metrics(y_true, y_pred, y_score, tag)

def evaluate_ensemble(models_dict, test_loader, tag="ensemble_AB"):
    for m in models_dict.values():
        m.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            probs_list = []
            for model in models_dict.values():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                probs = ordinal_probabilities(logits, NUM_CLASSES)
                probs_list.append(probs)

            avg_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)
            preds = torch.argmax(avg_probs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(avg_probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_score = np.concatenate(all_probs)

    return compute_and_save_metrics(y_true, y_pred, y_score, tag)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, dataset_sizes = get_dataloaders()
    dataloaders = {"train": train_loader, "val": val_loader}

    class_weights_tensor = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float32, device=device)

    trained_models = {}

    for backbone in BACKBONES:
        print(f"\n========== Training backbone: {backbone} ==========")
        model = build_model(backbone, NUM_THRESHOLDS, mode=MODE).to(device)

        lr = 5e-4 if MODE == "HEAD" else 1e-4
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model = train_model(
            model,
            dataloaders,
            dataset_sizes,
            optimizer,
            scheduler,
            class_weights_tensor=class_weights_tensor,
            num_epochs=EPOCHS,
            device=device,
            backbone_name=backbone,
        )

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{backbone}_ORD_ROI_AB_{MODE}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{backbone}] Saved best model to: {ckpt_path}")

        trained_models[backbone] = model
        _ = evaluate_single_model(model, test_loader, tag=f"{backbone}_AB")

    print("\n========== Evaluating ENSEMBLE (A+B) ==========")
    _ = evaluate_ensemble(trained_models, test_loader, tag="ensemble_AB")
