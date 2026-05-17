#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DATASET_ROOT = r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset\dataset_train_ready"
OUTPUT_DIR = r"C:\Users\zudae\Desktop\диплом1\model3\trained_model"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-3 clothing classes for one image.")
    parser.add_argument("--image", type=Path, default=None, help="Path to input image.")
    parser.add_argument("--dataset-root", type=Path, default=Path(DATASET_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir: Path = args.output_dir

    classes_path = output_dir / "class_names_used.json"
    model_path = output_dir / "best_model.pth"
    if not classes_path.exists() or not model_path.exists():
        raise FileNotFoundError("Missing best_model.pth or class_names_used.json in output dir")

    with classes_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names_used.json is invalid or empty")

    image_path = args.image
    if image_path is None:
        raise ValueError("Image path is not provided. Use --image <path_to_image>.")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    device = torch.device("cpu")
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(3, len(class_names))
    values, indices = torch.topk(probs, k=topk)

    print("Top-3 predictions:")
    for prob, idx in zip(values.tolist(), indices.tolist()):
        print(f"  {class_names[idx]}: {prob:.6f}")


if __name__ == "__main__":
    main()
