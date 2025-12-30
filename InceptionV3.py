"""
Train pretrained Inception v3 on knee OA X-ray dataset and evaluate on test set.

Hyperparameters:
- Batch size: 32
- Epochs: 70
- Optimizer: Adam
- LR: 5e-4  (IV3  : feature extraction, last layer only)
- LR: 1e-4  (FT_IV3: full fine-tuning)
- Loss: CrossEntropyLoss (sparse categorical CE)
"""

import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# from Dataset import KneeOADataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import numpy as np


# --------------------------- CONFIG ---------------------------

# Your dataset root
DATA_ROOT = r"E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5        # folders: 0,1,2,3,4
BATCH_SIZE = 32
EPOCHS = 70

# mode = "IV3"    -> feature extraction (only final layer trainable), lr = 5e-4
# mode = "FT_IV3" -> fine-tune whole network, lr = 1e-4
MODE = "IV3"          # change to "FT_IV3" if you want full fine-tuning


# ---------------------- DEVICE SELECTION ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------- DATASETS ---------------------------

# Inception v3 expects 299x299 images
input_size = 299

train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dir = os.path.join(DATA_ROOT, "train")
val_dir = os.path.join(DATA_ROOT, "val")
test_dir = os.path.join(DATA_ROOT, "test")

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=eval_transforms)
test_dataset  = datasets.ImageFolder(root=test_dir,  transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

print(f"Train samples: {dataset_sizes['train']}")
print(f"Val samples:   {dataset_sizes['val']}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Classes:       {train_dataset.classes}")


# ------------------------- MODEL ------------------------------

def build_inception_v3(num_classes, mode="IV3"):
    """
    mode = "IV3":    feature extraction (only final fc layer is trainable)
    mode = "FT_IV3": fine-tune entire network
    """
    model = models.inception_v3(pretrained=True, aux_logits=True)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Replace auxiliary classifier head
    if model.aux_logits:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    if mode == "IV3":
        # freeze all params
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze final layers
        for p in model.fc.parameters():
            p.requires_grad = True
        if model.aux_logits:
            for p in model.AuxLogits.parameters():
                p.requires_grad = True

    elif mode == "FT_IV3":
        # full fine-tuning
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("mode must be 'IV3' or 'FT_IV3'")

    return model


model = build_inception_v3(NUM_CLASSES, mode=MODE).to(device)

# --------------------- LOSS & OPTIMIZER ----------------------

criterion = nn.CrossEntropyLoss()

if MODE == "IV3":
    lr = 5e-4
else:  # FT_IV3
    lr = 1e-4

params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_update, lr=lr)

print(f"Training mode: {MODE}")
print(f"Learnable parameters: {len(params_to_update)}")
print(f"Learning rate: {lr}")


# ------------------------ TRAIN LOOP -------------------------

def train_model(model, dataloaders, criterion, optimizer,
                num_epochs=25, device="cpu"):
    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

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
                    # Inception v3 returns (main_output, aux_output) in train mode
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

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed/60:.1f}m")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# ---------------------- TEST EVALUATION ----------------------

def evaluate_on_test(model, test_loader, device="cpu"):
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
            # In eval mode, inception usually returns just outputs,
            # but handle tuple just in case.
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

    # Accuracy, Precision, Recall (Sensitivity), F1
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix for specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    # specificity per class
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

    # AUROC (macro, one-vs-rest)
    try:
        auroc = roc_auc_score(
            y_true, y_score, multi_class="ovr", average="macro"
        )
    except ValueError:
        auroc = float("nan")  # in case of a degenerate test set

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


# --------------------------- RUN -----------------------------

if __name__ == "__main__":
    # train and select best val model
    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=EPOCHS,
        device=device,
    )

    # save best model
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"inceptionv3_{MODE}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model saved to: {ckpt_path}")

    # evaluate on test set
    metrics = evaluate_on_test(model, test_loader, device=device)
