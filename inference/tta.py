import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.config import CFG

# ===============================
# TRAIN TRANSFORMS (REGULARIZED)
# ===============================
train_tfms = transforms.Compose([
    # Slightly larger resize to allow crop jitter
    transforms.Resize((CFG.img_size + 32, CFG.img_size + 32)),

    # Geometry (MRI-safe)
    transforms.RandomResizedCrop(
        CFG.img_size,
        scale=(0.85, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    # Intensity (MRI-safe, NO color semantics)
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.15,
        saturation=0.0,
        hue=0.0
    ),

    transforms.ToTensor(),

    # Scanner noise simulation
    transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),

    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ===============================
# VAL / TEST TRANSFORMS
# ===============================
val_tfms = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.transform = transform
        self.samples = []

        self.classes = sorted(
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            folder = os.path.join(images_dir, cls)
            for img_name in os.listdir(folder):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(folder, img_name), self.class_to_idx[cls])
                    )

        print(f"[Dataset] {len(self.samples)} images from {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def get_class_counts(self):
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for _, y in self.samples:
            counts[y] += 1
        return counts


def get_dataloaders():
    train_ds = BrainTumorDataset(
        os.path.join(CFG.data_root, "train", "images"),
        transform=train_tfms
    )
    val_ds = BrainTumorDataset(
        os.path.join(CFG.data_root, "val", "images"),
        transform=val_tfms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(CFG.device == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(CFG.device == "cuda")
    )

    return train_loader, val_loader, train_ds.classes, train_ds.get_class_counts()
