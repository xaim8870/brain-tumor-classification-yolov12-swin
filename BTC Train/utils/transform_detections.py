import cv2
import torch
import numpy as np
from utils.mri_preprocess import brain_margin_crop
from utils.yolo_bbox_utils import adjust_yolo_boxes


class MRIDetectionTransform:
    def __init__(self, img_size=512, mean=None, std=None):
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes):
        """
        image: np.ndarray (H, W, C)
        boxes: list of (cls, xc, yc, w, h)
        """
        orig_h, orig_w = image.shape[:2]

        # 1️⃣ Margin crop
        cropped, crop_box = brain_margin_crop(image)

        # 2️⃣ Adjust boxes
        boxes = adjust_yolo_boxes(
            boxes,
            crop_box,
            orig_w,
            orig_h
        )

        # 3️⃣ Resize to 512×512
        resized = cv2.resize(
            cropped,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR
        )

        # 4️⃣ Convert to tensor
        img = resized.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # 5️⃣ Normalize (after resize)
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean).view(3, 1, 1)
            std = torch.tensor(self.std).view(3, 1, 1)
            img = (img - mean) / std

        return img, boxes
