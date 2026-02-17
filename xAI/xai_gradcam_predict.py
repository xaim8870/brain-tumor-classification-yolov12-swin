# =========================================================
# Grad-CAM XAI for YOLOv12-Swin Classifier (TEST SET)
# =========================================================

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# FIX PROJECT ROOT (important)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.yolov12_swin_classifier import YOLOv12SwinClassifier
from utils.augmentations import val_transforms

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
TEST_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\merged_dataset_v4\images\test"
CHECKPOINT = r"D:\Brain-Tumor (2)\brain_tumor_classification\Results\checkpoints\best.pt"
OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\xai_gradcam"

os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = YOLOv12SwinClassifier(num_classes=len(CLASSES)).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---------------------------------------------------------
# HOOK FOR GRAD-CAM (last conv layer)
# ---------------------------------------------------------
target_layer = model.merge_conv
activations = []
gradients = []

def forward_hook(_, __, output):
    activations.append(output.detach())

def backward_hook(_, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ---------------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------------
transform = val_transforms(img_size=512)

def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img, tensor

# ---------------------------------------------------------
# GRAD-CAM COMPUTATION
# ---------------------------------------------------------
def compute_gradcam(img_tensor, class_idx):
    activations.clear()
    gradients.clear()

    logits = model(img_tensor)
    score = logits[:, class_idx]
    model.zero_grad()
    score.backward()

    A = activations[0]     # (1, C, H, W)
    G = gradients[0]       # (1, C, H, W)

    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1).squeeze()
    cam = F.relu(cam)

    cam = cam.cpu().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    return cam

# ---------------------------------------------------------
# SAVE VISUALIZATION
# ---------------------------------------------------------
def save_cam(orig_img, cam, pred_name, conf, out_path):
    img = np.array(orig_img)
    h, w, _ = img.shape
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(4.8, 4.8))
    plt.imshow(overlay)
    plt.title(f"{pred_name} ({conf:.2f}%)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------------------------------------------------------
# RUN ON TEST SET
# ---------------------------------------------------------
print("🔍 Running Grad-CAM on TEST images...")

for cls in CLASSES:
    cls_dir = os.path.join(TEST_DIR, cls)
    out_cls = os.path.join(OUT_DIR, cls)
    os.makedirs(out_cls, exist_ok=True)

    images = sorted(os.listdir(cls_dir))[:5]  # take first 5 per class (paper-friendly)

    for img_name in images:
        img_path = os.path.join(cls_dir, img_name)
        orig_img, img_tensor = load_image(img_path)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        pred_name = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100

        cam = compute_gradcam(img_tensor, pred_idx)

        out_path = os.path.join(
            out_cls,
            img_name.replace(".jpg", "_gradcam.png").replace(".png", "_gradcam.png")
        )

        save_cam(orig_img, cam, pred_name, confidence, out_path)

        print(f"✅ {cls}/{img_name} → {pred_name} ({confidence:.2f}%)")

print("\n🎯 Grad-CAM XAI completed. Results saved to:")
print(OUT_DIR)
