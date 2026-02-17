import torch
from torchvision import transforms
from PIL import Image
import os

IMG_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Robofolo-Dataset\images\train"

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

mean = torch.zeros(3)
std = torch.zeros(3)
total = 0

for cls in os.listdir(IMG_ROOT):
    for f in os.listdir(os.path.join(IMG_ROOT, cls)):
        img = Image.open(os.path.join(IMG_ROOT, cls, f)).convert("RGB")
        t = transform(img)
        mean += t.mean(dim=(1,2))
        std += t.std(dim=(1,2))
        total += 1

mean /= total
std /= total

print("Mean:", mean.tolist())
print("Std :", std.tolist())
