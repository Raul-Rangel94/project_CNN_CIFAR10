# Project CNN-CIFAR10

## 📌 Overview

This project explores the progression from a basic Convolutional Neural Network (CNN) to a Residual Network (ResNet-like architecture) for image classification on CIFAR-10.

The goal was not only to improve accuracy, but to understand how architectural decisions, regularization, and data augmentation impact model performance.

---

## 🚀 Results

| Model            | Validation Accuracy |
|------------------|-------------------|
| Basic CNN        | ~59%              |
| + BatchNorm      | ~61%              |
| + Dropout        | ~63%              |
| + Augmentation   | ~68%              |
| + ResNet-lite    | 88.58%            |
| + Full Pipeline  | **91.5%**         |

---

## 🧠 Key Learnings

- Increasing depth alone does not guarantee better performance  
- Proper regularization (Dropout, Label Smoothing) improves generalization  
- Data augmentation is critical for robustness  
- Architectural changes (ResNet) unlock major performance gains  

---

## 🏗️ Architecture

### 🔹 Baseline CNN
- 3 convolutional layers  
- Batch Normalization  
- Dropout  
- Global Average Pooling  

### 🔹 ResNet-lite
- Residual blocks with skip connections  
- Progressive channel expansion (32 → 64 → 128)  
- Downsampling via strided convolutions  
- Global Average Pooling  

---

## ⚙️ Training Setup

- **Optimizer:** Adam  
- **Loss:** CrossEntropyLoss + Label Smoothing  
- **Data Augmentation:**
  - RandomCrop  
  - HorizontalFlip  
  - ColorJitter  
  - Rotation  
- **Epochs:** 20–40  
- **Device:** GPU (Google Colab)  

---

## 📊 Observations

- Early models suffered from underfitting  
- Adding BatchNorm increased learning capacity  
- Dropout controlled overfitting  
- Strong augmentation improved generalization  
- Residual connections significantly improved feature learning  

---

## 📊 Results & Model Behavior

### 🎯 Final Model

The final model combines:

- ResNet-lite architecture  
- Data augmentation  
- Label smoothing  
- Mixup regularization  
- Cosine learning rate scheduler  

Achieving:

- **Validation Accuracy:** ~91.5%  
- Strong generalization beyond CIFAR-10  

---

### 🧪 Real-World Inference

The model was evaluated on external images (not part of CIFAR-10):

#### ✈️ Airplane
- Prediction: `airplane (88%)`  
- Correct classification with strong confidence  

#### 🚗 Automobile
- Prediction: `automobile (96%)`  
- Very confident and clean prediction  

#### 🐶🐱 Multi-object Image
- Prediction: `dog (79%)`, `cat (18%)`  
- Model captures multiple concepts despite single-label constraint  

#### 🐶 Ambiguous Case (Dog with leash)
- Prediction: `dog (56%)`, `deer (37%)`  
- Indicates confusion based on structural features  

#### 🐱 Clear Object
- Prediction: `cat (98%)`  
- High confidence on simple, centered inputs  

---

### 🧠 Model Behavior Insights

- The model generalizes well to real-world images  
- Predictions are generally well-calibrated  
- Errors are interpretable, not random  
- Strong reliance on **shape-based features (shape bias)**  

---

### ⚠️ Failure Cases

- Confusion between structurally similar classes (dog vs deer)  
- Sensitivity to object shape and background context  
- Multi-object scenes introduce ambiguity  

---

### 🚀 Key Takeaways

- Performance gains came from training strategy, not just architecture  
- Regularization techniques significantly improved generalization  
- Model behavior analysis is as important as raw accuracy  

---

## 🔥 Conclusion

This project highlights that improving deep learning models is not just about increasing complexity, but about understanding how training strategies, regularization, and data influence model behavior.

Beyond achieving high accuracy, analyzing model predictions on real-world data revealed important insights about generalization, calibration, and failure modes.

---

## 📌 Next Steps

- Learning rate schedulers (Cosine Annealing tuning)  
- Mixup / CutMix refinement  
- Deeper ResNet architectures  
- Transfer learning  

---

## 📁 Project Structure

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