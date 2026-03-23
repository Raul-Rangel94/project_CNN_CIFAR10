import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from src.data.transforms import get_test_transforms, get_train_transforms


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        transform=get_train_transforms(mean, std),
        download=True,
    )
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        transform=get_test_transforms(mean, std),
        download=True,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
