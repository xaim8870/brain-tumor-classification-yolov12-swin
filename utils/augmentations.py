# utils/augmentations.py
from torchvision import transforms

# ✅ Figshare 3-class dataset normalization (computed on train split)
MEAN = [0.18592512573200864, 0.18592512573200864, 0.18592512573200864]
STD  = [0.1773068168916504, 0.1773068168916504, 0.1773068168916504]


def train_transforms(img_size: int = 512):
    return transforms.Compose([
        # ❌ NO RandomResizedCrop (kills small tumors / ROI)

        transforms.Resize((img_size, img_size), antialias=True),

        # ✅ SAFE MRI augmentations
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomAffine(
            degrees=10,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            shear=0.0
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),

        # very light erasing (optional)
        transforms.RandomErasing(
            p=0.05,
            scale=(0.01, 0.04),
            ratio=(0.5, 2.0),
            value="random"
        ),
    ])


def val_transforms(img_size: int = 512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
