# cifar10-cnn

Estructura base de proyecto para clasificacion de CIFAR-10 con CNN en PyTorch.

## Estructura

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

## Uso

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Entrenar:

```bash
python -m src.train.train --config configs/config.yaml
```
