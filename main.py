import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.CNN_baseline import CNNBaseline
from utils.train import train
from utils.eval import evaluate
import config


def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def main():
    device = torch.device(config.DEVICE)

    # Data
    train_loader, test_loader = get_data_loaders()

    # Model
    model = CNNBaseline().to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Training loop
    best_accuracy = 0

    for epoch in range(config.EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        if (epoch + 1) % 2 == 0:
            accuracy = evaluate(model, test_loader, device)

            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {accuracy:.2f}%")

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "best_model.pth")
        else:
            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f}")

    print(f"\nBest Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()