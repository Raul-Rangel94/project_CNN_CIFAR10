import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn, optim

from src.data.dataset import get_cifar10_dataloaders
from src.models.cnn import build_model
from src.train.eval import evaluate
from src.utils.metrics import batch_accuracy
import torch.optim.lr_scheduler as lr_scheduler


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


def append_training_log(log_path: Path, run_id: str, epoch: int, total_epochs: int, train_metrics: dict, val_metrics: dict):
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)

        if not file_exists:
            writer.writerow(
                [
                    "run_id",
                    "epoch",
                    "total_epochs",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                ]
            )

        writer.writerow(
            [
                run_id,
                epoch,
                total_epochs,
                f"{train_metrics['loss']:.4f}",
                f"{train_metrics['accuracy']:.2f}",
                f"{val_metrics['loss']:.4f}",
                f"{val_metrics['accuracy']:.2f}",
            ]
        )


def append_training_summary(
    summary_path: Path,
    run_id: str,
    model_name: str,
    total_epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    uses_batch_norm: bool,
    dropout_rate: float,
    best_accuracy: float,
    best_epoch: int,
    final_train_metrics: dict,
    final_val_metrics: dict,
):
    file_exists = summary_path.exists()

    with open(summary_path, "a", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)

        if not file_exists:
            writer.writerow(
                [
                    "run_id",
                    "model_name",
                    "total_epochs",
                    "learning_rate",
                    "batch_size",
                    "weight_decay",
                    "uses_batch_norm",
                    "dropout_rate",
                    "best_val_acc",
                    "best_epoch",
                    "final_train_loss",
                    "final_train_acc",
                    "final_val_loss",
                    "final_val_acc",
                ]
            )

        writer.writerow(
            [
                run_id,
                model_name,
                total_epochs,
                learning_rate,
                batch_size,
                weight_decay,
                uses_batch_norm,
                f"{dropout_rate:.2f}",
                f"{best_accuracy:.2f}",
                best_epoch,
                f"{final_train_metrics['loss']:.4f}",
                f"{final_train_metrics['accuracy']:.2f}",
                f"{final_val_metrics['loss']:.4f}",
                f"{final_val_metrics['accuracy']:.2f}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=cfg["dataset"]["root"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        mean=tuple(cfg["dataset"]["mean"]),
        std=tuple(cfg["dataset"]["std"]),
    )

    model = build_model(
        model_name=model_name,
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.00)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    # Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs
    )
    models_dir, logs_dir = ensure_outputs(cfg["paths"])
    best_accuracy = 0.0
    best_epoch = 0
    log_path = logs_dir / "train_log.csv"
    summary_path = logs_dir / "summary.csv"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_epochs = cfg["training"]["epochs"]
    final_train_metrics = None
    final_val_metrics = None

    print(f"Run {run_id} | Model: {model_name} | Device: {device}")

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, test_loader, device, criterion)
        final_train_metrics = train_metrics
        final_val_metrics = val_metrics

        # Step the scheduler at the end of each epoch.
        scheduler.step()

        append_training_log(log_path, run_id, epoch, total_epochs, train_metrics, val_metrics)

        print(
            f"Run {run_id} | "
            f"Epoch {epoch}/{total_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), models_dir / "best_model.pth")

    append_training_summary(
        summary_path,
        run_id,
        model_name,
        total_epochs,
        cfg["optimizer"]["lr"],
        cfg["training"]["batch_size"],
        cfg["optimizer"]["weight_decay"],
        any(isinstance(module, nn.BatchNorm2d) for module in model.modules()),
        getattr(getattr(model, "dropout", None), "p", 0.0),
        best_accuracy,
        best_epoch,
        final_train_metrics,
        final_val_metrics,
    )

    torch.save(model.state_dict(), models_dir / "last_model.pth")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
