import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # skip connection
        return F.relu(out)
    
class Cifar10ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Entrada
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)

        # Bloques residuales
        self.layer1 = ResidualBlock(32)
        self.layer2 = ResidualBlock(32)

        # Downsample
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn_down1 = nn.BatchNorm2d(64)

        self.layer3 = ResidualBlock(64)
        self.layer4 = ResidualBlock(64)

        # Downsample
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn_down2 = nn.BatchNorm2d(128)

        self.layer5 = ResidualBlock(128)
        self.layer6 = ResidualBlock(128)

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))

        x = self.layer1(x)
        x = self.layer2(x)

        x = F.relu(self.bn_down1(self.down1(x)))

        x = self.layer3(x)
        x = self.layer4(x)

        x = F.relu(self.bn_down2(self.down2(x)))

        x = self.layer5(x)
        x = self.layer6(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return self.fc(x)


class Cifar10ResNetLite(Cifar10ResNet):
    pass


MODEL_REGISTRY = {
    "cnn": Cifar10CNN,
    "resnet_lite": Cifar10ResNetLite,
}


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    model_key = model_name.lower()

    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Modelo '{model_name}' no soportado. Modelos disponibles: {available_models}"
        )

    return MODEL_REGISTRY[model_key](num_classes=num_classes)
