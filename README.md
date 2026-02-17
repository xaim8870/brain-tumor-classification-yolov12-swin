# Brain Tumour MRI Classification (YOLO-inspired YOLOv12 × Swin Transformer)

A hybrid deep learning pipeline for **brain tumour classification from MRI scans**, built using a **YOLO-inspired YOLOv12 backbone combined with a Swin Transformer** to improve feature learning for medical imaging.  
This repository supports both:

- **3-class brain tumour classification** (Figshare dataset)
- **4-class brain tumour classification** (custom merged MRI dataset hosted on Hugging Face)

✅ Models and datasets are published on **Hugging Face**.  
✅ A separate GitHub repository provides a **deployed UI** so visitors can test predictions in the browser.

---

## 🔍 Project Highlights (SEO Keywords)
- Brain tumour MRI classification using deep learning
- YOLO-inspired YOLOv12 + Swin Transformer hybrid architecture
- 3-class and 4-class MRI multi-class classification
- PyTorch training with modern evaluation metrics
- Hugging Face model + dataset hosting
- Deployed inference UI for real-time testing

---

## 📦 What’s Included
- Training and validation pipeline (PyTorch)
- Multi-class classification (3-class + 4-class)
- Evaluation metrics (e.g., accuracy, precision, recall, F1, confusion matrix)
- Reproducible project structure for experiments and results
- Instructions to run training and inference locally

---

## 🧠 Datasets

### 1) 3-Class Dataset (Figshare)
We trained a **3-class MRI classification** model using an open dataset from **Figshare**.

- Classes: **3**
- Source: Figshare  


### 2) 4-Class Dataset (Merged + Hosted on Hugging Face)
We also trained a **4-class model** by **merging MRI scans from multiple sources** into a unified dataset and publishing it on **Hugging Face**.

- Classes: **4**
- Source: Custom merged dataset (multiple MRI datasets)
- Link: https://huggingface.co/datasets/usamaJabar/brain-tumor-mri-classification-merged

> Note: Always review each dataset’s licence and usage restrictions before commercial or clinical use.

---

## 🤖 Trained Models (Hugging Face)
All trained checkpoints are published on Hugging Face:

- **3-class model**: <ADD_HUGGINGFACE_MODEL_3CLASS_LINK>
- **4-class model**: <ADD_HUGGINGFACE_MODEL_4CLASS_LINK>

Optional:
* Model card includes training details, metrics, and intended use.

---

## 🌐 Live Demo UI (Separate Repository)
A deployed UI is available in another GitHub repository where users can upload MRI images and test the model predictions:

- UI Repository: <ADD_UI_GITHUB_REPO_LINK>
- Live Demo (if deployed): <ADD_LIVE_DEMO_LINK>

---

## ⚙️ Installation

### Requirements
- Python 3.10+ (recommended)
- PyTorch (CUDA optional)
- Common ML packages (NumPy, OpenCV, Matplotlib, etc.)

### Setup (Windows / Linux / macOS)
```bash
git clone https://github.com/<your-username>/brain-tumor-classification-yolov12-swin.git
cd brain-tumor-classification-yolov12-swin

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

```
### Results
<img width="1920" height="1200" alt="accuracy" src="https://github.com/user-attachments/assets/d0f59cfd-4ae6-4b08-b7bb-9738b9924332" />

<img width="1920" height="1200" alt="loss" src="https://github.com/user-attachments/assets/edf5b23a-9787-4893-9541-6b495188e8aa" />

