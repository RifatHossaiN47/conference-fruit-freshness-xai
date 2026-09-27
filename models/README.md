# Pre-Trained Model Weights

This directory contains configuration and instructions for the best-performing multi-task model: **ResNet152V2 Multi-Task Classifier**.

---

## Model Specifications

| Specification | Value |
| :--- | :--- |
| **Architecture** | ResNet152V2 Backbone + Dual Task Heads |
| **Total Parameters** | 59,564,682 (~59.6 M) |
| **Input Shape** | 224 × 224 × 3 (RGB) |
| **Weight File Name** | `ResNet152.keras` |
| **File Size** | ~348.7 MB (365,698,773 bytes) |
| **Grad-CAM Target Layer** | `conv5_block3_3_conv` (final conv feature map: 7×7×2048) |
| **Fruit Accuracy (Test)** | **99.64%** (9 Fruit Varieties) |
| **Freshness Accuracy (Test)** | **97.27%** (Fresh vs. Rotten) |
| **Combined Accuracy (Test)** | **98.45%** |
| **Freshness ROC-AUC** | **0.998** |

---

## ⚠️ GitHub 100 MB File Limit & Hosting

GitHub enforces a strict **100 MB maximum file size limit** for standard Git commits. Because `ResNet152.keras` is **348.7 MB**, it is excluded from Git tracking via `.gitignore` to prevent push rejections.

### How to Host and Provide the Weights:

### Option A: GitHub Releases (Recommended for simplicity)
1. In your GitHub repository, navigate to **Releases** > **Create a new release** (e.g., tag `v1.0.0`).
2. Attach `ResNet152.keras` as a binary asset (GitHub Releases allows assets up to 2 GB).
3. Users can download it directly:
   ```bash
   # Download into the models directory
   curl -L -o models/ResNet152.keras "https://github.com/RifatHossaiN47/fruit-freshness-multitask-xai/releases/download/v1.0.0/ResNet152.keras"
   ```

### Option B: Hugging Face Model Hub
You can upload the model directly to your Hugging Face Space or Model repository:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="RifatHossaiN47/fruit-freshness-resnet152", filename="ResNet152.keras", local_dir="models")
```

### Option C: Git LFS (Large File Storage)
If you prefer keeping weights inside the repository history:
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/ResNet152.keras
git commit -m "feat: add pretrained ResNet152 weights via Git LFS"
```
