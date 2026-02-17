import matplotlib.pyplot as plt
import os

IMG_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Robofolo-Dataset\images\train"

classes = []
counts = []

for cls in os.listdir(IMG_ROOT):
    classes.append(cls)
    counts.append(len(os.listdir(os.path.join(IMG_ROOT, cls))))

plt.bar(classes, counts)
plt.title("Training Class Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
