"""
Grad-CAM visualization for ViT-L/16 CORAL knee KL-grade model.

- Loads trained ViT-L/16 CORAL checkpoint.
- Computes Grad-CAM for a chosen target KL grade.
- Saves overlay heatmaps on the original 224x224 knee crop.

Tested with torchvision>=0.15 and PyTorch>=1.13.
Adjust checkpoint path and image list before running.
"""

import os
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5  # KL0..4  -> thresholds K-1 = 4
IMG_SIZE = 224

CHECKPOINT_PATH = "ViT_L16_CORAL_fold4_best.pth"  # <-- change to your checkpoint

# List your images here (you can start with your 5 examples)
IMAGE_PATHS: List[str] = [
    r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\train\0\9001695L.png",
    r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\train\1\9000622L.png",
    r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\train\2\9000099R.png",
    r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\train\3\9000099L.png",
    r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224\train\4\9025994L.png",
]


# Target KL grade for Grad-CAM for each image above (same length as IMAGE_PATHS)
TARGET_GRADES: List[int] = [0, 1, 2, 3, 4]

OUTPUT_DIR = "gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# MODEL: ViT-L/16 + CORAL
# -------------------------------------------------------------------------

class ViT_CORAL_L16(nn.Module):
    """
    ViT-L/16 backbone with CORAL ordinal head: predicts K-1 thresholds.
    This should match the architecture used for training.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        num_thresholds = num_classes - 1

        # Try new torchvision weights API first, fall back to pretrained=True
        try:
            vit = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except TypeError:
            vit = models.vit_l_16(pretrained=pretrained)

        embed_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()   # remove classification head
        self.backbone = vit
        self.classifier = nn.Linear(embed_dim, num_thresholds)

    def forward(self, x):
        feats = self.backbone(x)         # [B, embed_dim]
        logits = self.classifier(feats)  # [B, K-1]
        return logits


# -------------------------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------------------------

transform_input = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> (torch.Tensor, np.ndarray):
    """
    Load an image file, return:
      - preprocessed tensor for model [1,3,H,W]
      - original image as numpy array [H,W,3] in [0,1] for plotting
    """
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = transform_input(img_resized).unsqueeze(0)  # [1,3,H,W]
    return img_tensor, img_np


# -------------------------------------------------------------------------
# Grad-CAM for ViT
# -------------------------------------------------------------------------

class ViTGradCAM:
    """
    Grad-CAM implementation for VisionTransformer:
    - hooks the output of the last encoder block (tokens)
    - converts tokens (except CLS) into a 2D map
    """

    def __init__(self, model: ViT_CORAL_L16):
        self.model = model
        self.model.eval()

        self.activations = None  # [B, tokens, dim]
        self.gradients = None    # [B, tokens, dim]

        # Hook on the last encoder block's output (after ln_2)
        last_block = self.model.backbone.encoder.layers[-1].ln_2

        def forward_hook(module, input, output):
            # output: [B, tokens, dim]
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: [B, tokens, dim]
            self.gradients = grad_output[0]

        last_block.register_forward_hook(forward_hook)
        last_block.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_grade: int) -> np.ndarray:
        """
        input_tensor: [1,3,H,W]
        target_grade: 0..4
        returns heatmap in [0,1] of shape [H,W]
        """

        input_tensor = input_tensor.to(DEVICE)

        # Forward
        self.model.zero_grad()
        logits = self.model(input_tensor)  # [1, K-1]

        # For CORAL, we have K-1 thresholds. Choose one associated with grade.
        # Simple rule: target index = clip(grade-1, 0, K-2)
        num_thresholds = logits.shape[1]
        target_idx = min(max(target_grade - 1, 0), num_thresholds - 1)

        target_logit = logits[0, target_idx]
        target_logit.backward()

        # activations & gradients: [1, tokens, dim]
        act = self.activations          # [1, T, D]
        grad = self.gradients           # [1, T, D]

        # Remove CLS token (first token), keep patch tokens only
        act = act[:, 1:, :]             # [1, T-1, D]
        grad = grad[:, 1:, :]           # [1, T-1, D]

        b, tokens, dim = act.shape
        side = int(math.sqrt(tokens))   # e.g., 14x14 tokens for 224x224 with 16x16 patches
        if side * side != tokens:
            raise RuntimeError(f"Unexpected tokens={tokens}; cannot reshape to square.")

        # Compute channel-wise weights by global average pooling of gradients
        # grad: [1, T, D] -> weights: [1, D]
        weights = grad.mean(dim=1)      # [1, D]

        # Weighted sum of activations -> [1, T]
        cam_tokens = (act * weights.unsqueeze(1)).sum(dim=-1)  # [1, T]
        cam = cam_tokens.reshape(1, 1, side, side)             # [1,1,S,S]

        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


# -------------------------------------------------------------------------
# VISUALIZATION HELPERS
# -------------------------------------------------------------------------

def overlay_heatmap(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a jet heatmap onto a grayscale or RGB image.
    img_np: [H,W] or [H,W,3] in [0,1]
    cam:    [H,W] in [0,1]
    alpha: blending ratio
    """
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)

    # Colormap: use matplotlib jet
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]  # [H,W,3]

    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_gradcam_figure(img_np: np.ndarray, cam: np.ndarray,
                        out_path: str, title: str = ""):
    """
    Save original image, heatmap, and overlay in a single 1x3 figure.
    """
    overlay = overlay_heatmap(img_np, cam, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    im1 = axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    # Load model and checkpoint
    print("Loading model...")
    model = ViT_CORAL_L16(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # If you saved with {"model": state_dict, ...}:
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Checkpoint loaded.")

    gradcam = ViTGradCAM(model)

    # Loop over all images
    for img_path, grade in zip(IMAGE_PATHS, TARGET_GRADES):
        if not os.path.isfile(img_path):
            print(f"[WARN] image not found, skipping: {img_path}")
            continue

        img_tensor, img_np = load_image(img_path)
        cam = gradcam.generate(img_tensor, target_grade=grade)

        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR,
                                f"{fname}_gradcam_KL{grade}.png")
        title = f"Grad-CAM (target grade KL{grade})"
        save_gradcam_figure(img_np, cam, out_path, title=title)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
