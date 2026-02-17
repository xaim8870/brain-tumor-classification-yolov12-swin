import cv2
import numpy as np


def brain_margin_crop(image, threshold=10, pad_ratio=0.05):
    """
    Removes black margins from MRI using simple intensity thresholding.
    Safe for MRI (no random cropping).

    Args:
        image (np.ndarray): HxWxC or HxW
        threshold (int): pixel intensity threshold
        pad_ratio (float): padding after crop (percentage of size)

    Returns:
        cropped_image, crop_box (x1, y1, x2, y2)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binary mask (brain ≠ black background)
    mask = gray > threshold

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        # fallback: no crop
        h, w = gray.shape
        return image, (0, 0, w, h)

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    h, w = gray.shape
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = image[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)
