"""
Inception V3 for Knee OA severity (5 classes: 0..4)

- Uses full images with CLAHE preprocessing.
- Balances training with WeightedRandomSampler instead of discarding data.
- Strong augmentation on train, plain transforms on val/test.
- Supports IV3 (feature extraction) and FT_IV3 (full fine tuning).
- Reports accuracy, precision, sensitivity, F1, specificity, AUROC on test.
- Saves metrics to CSV, plots ROC curves, and generates Grad-CAM heatmaps.

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
import csv
import time
from copy import deepcopy

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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
    roc_curve,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_ROOT = r"E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL299"

NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 70

# MODE:
#   "IV3"    - feature extraction (only classifier layer learns), lr = 5e-4
#   "FT_IV3" - full fine tuning, lr = 1e-4
MODE = "FT_IV3"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# PREPROCESSING: CLAHE ON FULL IMAGE
# ---------------------------------------------------------------------

def clahe_preprocess(pil_img: Image.Image) -> Image.Image:
    """
    1. Convert to grayscale
    2. CLAHE for local contrast enhancement
    3. Convert back to 3-channel RGB PIL image
    """
    img_np = np.array(pil_img.convert("L"))  # grayscale
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(img_np)
    eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(eq_rgb)


# ---------------------------------------------------------------------
# DATASETS AND DATALOADERS
# ---------------------------------------------------------------------

def get_dataloaders():
    input_size = 299  # Inception V3 default

    train_transform = transforms.Compose([
        transforms.Lambda(clahe_preprocess),
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
        transforms.RandomErasing(p=0.25),
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(clahe_preprocess),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=test_dir,  transform=eval_transform)

    # WeightedRandomSampler to balance classes without dropping data
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    print("Train class counts:", class_counts.tolist())

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
    }

    print("Dataset sizes:")
    print("  train:", dataset_sizes["train"])
    print("  val:  ", dataset_sizes["val"])
    print("  test: ", len(test_dataset))
    print("  classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader, dataset_sizes, (
        train_dataset, val_dataset, test_dataset
    )


# ---------------------------------------------------------------------
# MODEL: INCEPTION V3 (NEW weights API)
# ---------------------------------------------------------------------

def build_inception_v3(num_classes, mode="IV3"):
    # Use modern weights API instead of pretrained=True
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(
        weights=weights,
        aux_logits=True
    )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if model.aux_logits:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    if mode == "IV3":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if model.aux_logits:
            for p in model.AuxLogits.parameters():
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
                criterion, optimizer, scheduler,
                num_epochs=25, device="cpu"):

    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 32)

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
                    if phase == "train" and model.training and model.aux_logits:
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        scheduler.step()

        # keep best val weights
        if epoch_acc > best_acc and phase == "val":
            best_acc = epoch_acc
            best_model_wts = deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed / 60:.1f} minutes")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ---------------------------------------------------------------------
# TEST EVALUATION + CSV + ROC
# ---------------------------------------------------------------------

def evaluate_on_test(model, test_loader, device="cpu", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = softmax(outputs)
            _, preds = torch.max(probs, 1)

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
            f"Class {i}: "
            f"Precision={per_class_prec[i]:.4f}, "
            f"Recall={per_class_rec[i]:.4f}, "
            f"F1={per_class_f1[i]:.4f}"
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

    # ---- Save overall metrics to CSV ----
    overall_path = os.path.join(out_dir, "overall_metrics.csv")
    with open(overall_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", acc])
        writer.writerow(["precision_macro", precision])
        writer.writerow(["recall_macro", recall])
        writer.writerow(["f1_macro", f1])
        writer.writerow(["specificity_macro", specificity])
        writer.writerow(["auroc_macro_ovr", auroc])

    # ---- Save per-class metrics to CSV ----
    per_class_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(per_class_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "specificity"])
        for i in range(NUM_CLASSES):
            writer.writerow([i,
                             float(per_class_prec[i]),
                             float(per_class_rec[i]),
                             float(per_class_f1[i]),
                             float(spec_per_class[i])])

    # ---- Save confusion matrix to CSV ----
    cm_path = os.path.join(out_dir, "confusion_matrix_299.csv")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    # ---- ROC curves per class ----
    plt.figure()
    for i in range(NUM_CLASSES):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    roc_fig_path = os.path.join(out_dir, "roc_curves_299.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "specificity_macro": specificity,
        "auroc_macro_ovr": auroc,
        "confusion_matrix": cm,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
        "per_class_specificity": spec_per_class,
    }


# ---------------------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer_name="Mixed_7c"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        target_layer = dict([*self.model.named_modules()])[target_layer_name]

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.cpu().numpy()
        return cam


def save_gradcam_examples(model, dataset, device, out_dir="gradcam_examples", num_examples=10):
    os.makedirs(out_dir, exist_ok=True)
    gradcam = GradCAM(model, target_layer_name="Mixed_7c")

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for idx in range(min(num_examples, len(dataset))):
        img_tensor, label = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        cam = gradcam.generate(inp)

        img_denorm = inv_normalize(img_tensor).clamp(0, 1)
        img_np = np.transpose(img_denorm.cpu().numpy(), (1, 2, 0))

        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.4 * heatmap / 255.0 + 0.6 * img_np)
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original (label {label})")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fname = os.path.join(out_dir, f"gradcam_example_{idx}_label_{label}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset_sizes, datasets = get_dataloaders()
    train_dataset, val_dataset, test_dataset = datasets
    dataloaders = {"train": train_loader, "val": val_loader}

    model = build_inception_v3(NUM_CLASSES, mode=MODE).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if MODE == "IV3":
        lr = 5e-4
    else:
        lr = 1e-4

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nMode: {MODE}, lr: {lr}, trainable params: {len(params_to_update)}")

    model = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        num_epochs=EPOCHS,
        device=device,
    )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"inceptionv3_{MODE}_weighted.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved best model to:", ckpt_path)

    _ = evaluate_on_test(model, test_loader, device=device, out_dir="results")

    # Grad-CAM visualizations for a few test images
    save_gradcam_examples(model, test_dataset, device, out_dir="gradcam_examples", num_examples=10)
