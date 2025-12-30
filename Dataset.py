import os
import numpy as np
import cv2
import pywt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ============================================================
#                    KNEE OA DATASET
# ============================================================
class KneeOADataset(Dataset):
    """
    Directory structure:
        root/
            train/
                0/
                1/
                2/
                3/
                4/
            val/
                0/
                1/
                2/
                3/
                4/
            test/
                0/
                1/
                2/
                3/
                4/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",           # 'train', 'val', 'test'
        size=(224, 224),
        base_transform=None,
        augment_classes=(1, 3, 4),      # class folders to augment
    ):
        super().__init__()

        assert split in ["train", "val", "test"]
        self.root_dir = root_dir
        self.split = split
        self.size = size
        self.base_transform = base_transform

        # classes for which to apply augmentation (only in train)
        self.augment_classes = set(augment_classes)

        # build image list and labels
        self.image_paths, self.labels = self._build_dataset()

        # augmentation transform: horizontal flip + zoom + translation
        # zoom + translation are done with RandomAffine
        self.augment_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=0,                 # no rotation
                    translate=(0.1, 0.1),      # up to 10% translation
                    scale=(0.9, 1.1),          # zoom in/out 10%
                ),
            ]
        )

    # ---------------------------------------------------------
    def _build_dataset(self):
        split_dir = os.path.join(self.root_dir, self.split)

        class_names = sorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )

        image_paths = []
        labels = []

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(int(class_name))  # folder name is the label

        return image_paths, labels

    # ---------------------------------------------------------
    # Wavelet + CLAHE preprocessing
    # ---------------------------------------------------------
    def _preprocess(self, img_np: np.ndarray) -> np.ndarray:
        # Ensure grayscale
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_np.copy()

        gray = gray.astype(np.float32) / 255.0

        # Wavelet denoising
        coeffs = pywt.wavedec2(gray, "db1", level=2)
        cA, detail = coeffs[0], coeffs[1:]

        cH, cV, cD = detail[0]
        sigma = np.median(np.abs(cH)) / 0.6745
        thresh = sigma * np.sqrt(2 * np.log(gray.size))

        new_detail = []
        for cH, cV, cD in detail:
            cH = pywt.threshold(cH, thresh, mode="soft")
            cV = pywt.threshold(cV, thresh, mode="soft")
            cD = pywt.threshold(cD, thresh, mode="soft")
            new_detail.append((cH, cV, cD))

        denoised = pywt.waverec2([cA] + new_detail, "db1")
        denoised = np.clip(denoised, 0, 1)
        denoised_uint8 = (denoised * 255).astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised_uint8)

        # Resize
        resized = cv2.resize(enhanced, self.size, interpolation=cv2.INTER_CUBIC)

        # Back to 3 channels for CNNs
        out = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return out

    # ---------------------------------------------------------
    def __len__(self):
        return len(self.image_paths)

    # ---------------------------------------------------------
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and convert to numpy
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Preprocess (wavelet + CLAHE + resize)
        img_np = self._preprocess(img_np)

        # Convert to PIL for torchvision transforms
        img_pil = Image.fromarray(img_np)

        # Apply augmentation only for train split and specified classes
        if self.split == "train" and label in self.augment_classes:
            img_pil = self.augment_transform(img_pil)

        # Base transform (e.g. ToTensor, Normalize)
        if self.base_transform is not None:
            img_tensor = self.base_transform(img_pil)
        else:
            img_tensor = T.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label).long()


# ============================================================
#                 VISUALIZATION HELPER
# ============================================================
def plot_batch(x, y, max_imgs=9):
    max_imgs = min(max_imgs, x.size(0))
    rows = cols = int(np.ceil(np.sqrt(max_imgs)))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(rows * cols):
        axes[i].axis("off")

    for i in range(max_imgs):
        img = x[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {y[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
#                        MAIN EXAMPLE
# ============================================================
if __name__ == "__main__":
    # Change this to your dataset root
    root_dir = "E:\\Knee-OsteoArthritis-severity-detection\\KneeXrayData\\KneeXrayData\\ClsKLData\\kneeKL224"   # e.g., "D:/Knee_OA_Data"
    img_size = (224, 224)
    batch_size = 8

    base_transform = T.Compose(
        [
            T.ToTensor(),
            # Optional: add normalization if you use ImageNet-pretrained models
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = KneeOADataset(
        root_dir=root_dir,
        split="train",
        size=img_size,
        base_transform=base_transform,
        augment_classes=(1, 3, 4),
    )

    val_dataset = KneeOADataset(
        root_dir=root_dir,
        split="val",
        size=img_size,
        base_transform=base_transform,
        augment_classes=(),  # no aug in val
    )

    test_dataset = KneeOADataset(
        root_dir=root_dir,
        split="test",
        size=img_size,
        base_transform=base_transform,
        augment_classes=(),  # no aug in test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train samples:", len(train_dataset))
    print("Val samples:  ", len(val_dataset))
    print("Test samples: ", len(test_dataset))

    # Visualize one augmented batch from train (folders 1,3,4 will be augmented)
    for xb, yb in train_loader:
        plot_batch(xb, yb, max_imgs=batch_size)
        break
