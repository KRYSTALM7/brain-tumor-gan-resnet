# Brain Tumor Detection — GAN + ResNet50

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
[![Paper](https://img.shields.io/badge/Paper-IET%202025-purple?style=flat-square)](https://doi.org/10.1049/PBPC076E_ch6)

A deep learning pipeline for binary brain tumor classification from MRI scans, combining **Generative Adversarial Networks (GAN)** for data augmentation with a **ResNet50** transfer learning classifier.

This project is an implementation based on the research published in:

> MV Sujan Kumar, Ganesh Khekare, Shashi Kant Gupta, and Sharnil Pandya. *Harnessing generative AI for enhanced brain tumor detection in clinical trials.* In **Generative AI Unleashed**, Chapter 6, pp. 83–101, IET, 2025. https://doi.org/10.1049/PBPC076E_ch6

---

## Overview

Brain tumor detection from MRI scans is a critical clinical task where dataset size is a constant bottleneck. This project addresses class imbalance and data scarcity using a GAN-based augmentation strategy, feeding synthetic MRI images alongside real ones into a frozen ResNet50 backbone for binary classification (Tumor / No Tumor).

---

## Architecture

```
Brain MRI Dataset (253 images)
         │
         ▼
  Data Augmentation
  (flips → 1,000 images/class)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
GAN Training  Real Images
(200 epochs)  (normalised)
    │         │
    ▼         │
Generator     │
Conv2DTranspose ×3
tanh output   │
    │         │
    ▼         ▼
Synthetic   Train Split
MRI Scans   (70%)
    │         │
    └────┬────┘
         │
         ▼
  Combined Training Set
  (Real + GAN-augmented)
         │
         ▼
  ResNet50 (frozen, ImageNet)
         │
    Flatten
         │
    Dense(1024) + Dropout(0.4)
         │
    Dense(1, sigmoid)
         │
         ▼
  Tumor / No Tumor
```

### GAN Architecture

| Component | Details |
|-----------|---------|
| Input | Latent vector (dim=100) |
| Generator | Dense → Reshape → Conv2DTranspose ×3 → tanh |
| Discriminator | Conv2D ×2 + LeakyReLU(0.2) + Dropout → Dense(1, sigmoid) |
| Output size | 32×32 grayscale images |
| Training | 200 epochs, label smoothing 0.9, Adam (lr=0.0002) |

### Classifier Architecture

| Component | Details |
|-----------|---------|
| Backbone | ResNet50 (frozen, ImageNet weights) |
| Input size | 256×256×3 |
| Head | Flatten → Dense(1024, ReLU) → Dropout(0.4) → Dense(1, sigmoid) |
| Training | 10 epochs, Adam, binary cross-entropy |
| Train/Test split | 70% / 30% |

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **98.87%** |
| Precision | 93.74% |
| Recall | 92.14% |
| F1 Score | 92.93% |
| AUC-ROC | 0.96 |

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

### ROC Curve

![ROC Curve](assets/roc_curve.png)

---

## Dataset

[Kaggle — Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

| Split | Normal | Tumor |
|-------|--------|-------|
| Raw | 98 | 155 |
| After augmentation | 1,000 | 1,000 |
| Train | — | 708 total |
| Test | — | 304 total |

> Dataset not included in this repository. Download and place images in `data/yes/` and `data/no/`.

---

## Project Structure

```
brain-tumor-gan-resnet/
├── app/
│   └── app.py              # Gradio inference app
├── assets/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── data/
│   ├── yes/                # Tumor MRI images (not included)
│   ├── no/                 # Normal MRI images (not included)
│   └── README.md
├── notebooks/
│   └── brain_tumor_detection.ipynb
├── outputs/
│   ├── models/             # Saved .keras models (not included)
│   └── plots/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── gan.py
│   ├── classifier.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
├── LICENSE
└── README.md
```

> Source code is intentionally omitted from this repository. The architecture, methodology, and results are documented here for portfolio and research reference.

---

## Citation

```bibtex
@inbook{doi:10.1049/PBPC076E_ch6,
  author    = {MV Sujan Kumar and Ganesh Khekare and Shashi Kant Gupta and Sharnil Pandya},
  title     = {Harnessing generative AI for enhanced brain tumor detection in clinical trials},
  booktitle = {Generative AI Unleashed},
  chapter   = {Chapter 6},
  pages     = {83--101},
  doi       = {10.1049/PBPC076E_ch6},
  url       = {https://digital-library.theiet.org/doi/abs/10.1049/PBPC076E_ch6},
  year      = {2025},
  publisher = {Institution of Engineering and Technology}
}
```

---

## License

This repository is licensed under the [MIT License](LICENSE).

The implementation is based on published research. All academic credit belongs to the original authors cited above.