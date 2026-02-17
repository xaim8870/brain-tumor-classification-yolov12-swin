import torch
from PIL import Image
from torchvision import transforms

MEAN = [0.17547428607940674, 0.17547376453876495, 0.17547328770160675]
STD  = [0.17224352061748505, 0.1722438931465149, 0.17224451899528503]

IMG_SIZE = 512

_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def preprocess_image(image_path: str, device="cpu"):
    img = Image.open(image_path).convert("RGB")
    x = _preprocess(img).unsqueeze(0)  # (1,C,H,W)
    return x.to(device)
