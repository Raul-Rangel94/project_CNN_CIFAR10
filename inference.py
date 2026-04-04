import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.models.cnn import build_model


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_inference_transforms(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_model(model_path: str, config_path: str, device: torch.device):
    cfg = load_config(config_path)
    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def tta_predict(model, image, device):
    image = image.to(device)
    flipped_image = torch.flip(image, dims=[3])

    with torch.no_grad():
        logits = model(image)
        flipped_logits = model(flipped_image)

        probs = torch.softmax(logits, dim=1)
        flipped_probs = torch.softmax(flipped_logits, dim=1)
        avg_probs = (probs + flipped_probs) / 2.0

    predicted_idx = torch.argmax(avg_probs, dim=1).item()
    return predicted_idx, avg_probs.squeeze(0).cpu()


def predict_image(image_path, model_path, config_path="configs/config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(model_path, config_path, device)

    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transforms(
        mean=tuple(cfg["dataset"]["mean"]),
        std=tuple(cfg["dataset"]["std"]),
    )
    image_tensor = transform(image).unsqueeze(0)

    predicted_idx, probabilities = tta_predict(model, image_tensor, device)
    top_probs, top_indices = torch.topk(probabilities, k=3)

    print(f"Predicted class: {CLASSES[predicted_idx]} ({predicted_idx})")
    print("Top-3 probabilities:")
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        print(f"  {CLASSES[idx]} ({idx}): {prob:.4f}")

    return predicted_idx, probabilities


def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 inference with TTA")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    predict_image(
        image_path=args.image,
        model_path=args.model,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
