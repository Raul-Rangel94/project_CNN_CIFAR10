import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch import nn, optim

from src.data.dataset import get_cifar10_dataloaders
from src.models.cnn import Cifar10CNN
from src.train.eval import evaluate
from src.utils.metrics import batch_accuracy


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += batch_accuracy(logits, labels)

    return {
        "loss": running_loss / len(loader),
        "accuracy": 100.0 * running_acc / len(loader),
    }


def ensure_outputs(paths_cfg):
    models_dir = Path(paths_cfg["models_dir"])
    logs_dir = Path(paths_cfg["logs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, logs_dir


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=cfg["dataset"]["root"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        mean=tuple(cfg["dataset"]["mean"]),
        std=tuple(cfg["dataset"]["std"]),
    )

    model = Cifar10CNN(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    models_dir, logs_dir = ensure_outputs(cfg["paths"])
    best_accuracy = 0.0
    log_path = logs_dir / "train_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, test_loader, device, criterion)

            writer.writerow(
                [
                    epoch,
                    f"{train_metrics['loss']:.4f}",
                    f"{train_metrics['accuracy']:.2f}",
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['accuracy']:.2f}",
                ]
            )

            print(
                f"Epoch {epoch}/{cfg['training']['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                torch.save(model.state_dict(), models_dir / "best_model.pth")

    torch.save(model.state_dict(), models_dir / "last_model.pth")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
