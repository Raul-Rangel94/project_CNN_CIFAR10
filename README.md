# Project CNN-CIFAR10
## 📌 Overview

This project explores the progression from a basic Convolutional Neural Network (CNN) to a Residual Network (ResNet-like architecture) for image classification on CIFAR-10.

The goal was not only to improve accuracy, but to understand how architectural decisions, regularization, and data augmentation impact model performance.

## 🚀 Results
Model	Validation Accuracy
Basic CNN	~59%
+ BatchNorm	~61%
+ Dropout	~63%
+ Augmentation	~68%
+ ResNet-lite	88.58%
## 🧠 Key Learnings
Increasing depth alone does not guarantee better performance
Proper regularization (Dropout, Label Smoothing) improves generalization
Data augmentation is critical for robustness
Architectural changes (ResNet) unlock major performance gains
## 🏗️ Architecture
🔹 Baseline CNN
3 convolutional layers
Batch Normalization
Dropout
Global Average Pooling
## 🔹 ResNet-lite
Residual blocks with skip connections
Progressive channel expansion (32 → 64 → 128)
Downsampling via strided convolutions
Global Average Pooling
## ⚙️ Training Setup
Optimizer: Adam
Loss: CrossEntropyLoss + Label Smoothing
Data Augmentation:
RandomCrop
HorizontalFlip
ColorJitter
Rotation
Epochs: 20
Device: GPU (Google Colab)
## 📊 Observations
Early models suffered from underfitting
Adding BatchNorm increased learning capacity
Dropout controlled overfitting
Strong augmentation improved generalization
Residual connections significantly improved feature learning

## 🔥 Conclusion

This project demonstrates how performance improvements in deep learning are not achieved through random tuning, but through systematic experimentation and understanding model behavior.

## 📌 Next Steps
Learning rate schedulers (Cosine Annealing)
Mixup / CutMix augmentation
Deeper ResNet architectures
Transfer learning

## Project Structure

```text
cifar10-cnn/
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_improved_model.ipynb
│   └── 03_experiments.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   └── cnn.py
│   ├── train/
│   │   ├── train.py
│   │   └── eval.py
│   └── utils/
│       └── metrics.py
├── configs/
│   └── config.yaml
├── outputs/
│   ├── models/
│   └── logs/
├── requirements.txt
└── README.md
```

## Utility

1. Instal dependencies:

```bash
pip install -r requirements.txt
```

2. Train:

```bash
python -m src.train.train --config configs/config.yaml
```
