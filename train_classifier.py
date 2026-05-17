#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

DATASET_ROOT = r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset\dataset_train_ready"
OUTPUT_DIR = r"C:\Users\zudae\Desktop\диплом1\model3\trained_model"
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FilteredImageFolder(Dataset):
    def __init__(self, root: Path, transform, used_classes: Sequence[str]):
        self.base = datasets.ImageFolder(str(root), transform=transform)
        self.used_classes = list(used_classes)
        self.class_to_idx_new = {c: i for i, c in enumerate(self.used_classes)}

        keep_old_indices = {
            self.base.class_to_idx[c]
            for c in self.used_classes
            if c in self.base.class_to_idx
        }

        self.samples: List[tuple[str, int]] = []
        for path, old_target in self.base.samples:
            if old_target in keep_old_indices:
                class_name = self.base.classes[old_target]
                self.samples.append((path, self.class_to_idx_new[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        sample = self.base.loader(path)
        if self.base.transform is not None:
            sample = self.base.transform(sample)
        return sample, target


@dataclass
class EpochResult:
    epoch: int
    stage: str
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clothing classifier (MobileNetV3 Small).")
    parser.add_argument("--dataset-root", type=Path, default=Path(DATASET_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=10)
    parser.add_argument("--epochs-stage2", type=int, default=10)
    parser.add_argument("--min-train-count", type=int, default=100)
    return parser.parse_args()


def load_stats_and_classes(dataset_root: Path, min_train_count: int):
    stats_path = dataset_root / "dataset_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"dataset_stats.csv not found: {stats_path}")

    rows = []
    with stats_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    used_classes: List[str] = []
    skipped_classes: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}

    for row in rows:
        cls = row["class_name"]
        train_count = int(float(row.get("train_count", 0) or 0))
        val_count = int(float(row.get("val_count", 0) or 0))
        counts[cls] = {"train": train_count, "val": val_count}
        if train_count >= min_train_count:
            used_classes.append(cls)
        else:
            skipped_classes.append(cls)

    return used_classes, skipped_classes, counts


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def freeze_all_features(model):
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_two_feature_blocks(model):
    for p in model.features.parameters():
        p.requires_grad = False
    children = list(model.features.children())
    for block in children[-2:]:
        for p in block.parameters():
            p.requires_grad = True


def set_classifier_trainable(model, trainable: bool = True):
    for p in model.classifier.parameters():
        p.requires_grad = trainable


def evaluate_and_collect(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())
    return y_true, y_pred


def save_training_log(log_rows: List[EpochResult], out_csv: Path):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "stage", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"],
        )
        writer.writeheader()
        for r in log_rows:
            writer.writerow(
                {
                    "epoch": r.epoch,
                    "stage": r.stage,
                    "train_loss": f"{r.train_loss:.6f}",
                    "train_accuracy": f"{r.train_acc:.6f}",
                    "val_loss": f"{r.val_loss:.6f}",
                    "val_accuracy": f"{r.val_acc:.6f}",
                    "learning_rate": f"{r.lr:.8f}",
                }
            )


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    dataset_root: Path = args.dataset_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    used_classes, skipped_classes, counts = load_stats_and_classes(dataset_root, args.min_train_count)
    if not used_classes:
        raise RuntimeError("No classes satisfy min_train_count. Nothing to train.")

    print("Used classes:", used_classes)
    print("Skipped classes:", skipped_classes)
    print("Counts for used classes:")
    for c in used_classes:
        print(f"  {c}: train={counts[c]['train']}, val={counts[c]['val']}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = FilteredImageFolder(dataset_root / "train", train_tf, used_classes)
    val_ds = FilteredImageFolder(dataset_root / "val", val_tf, used_classes)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty filtered datasets: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(used_classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    logs: List[EpochResult] = []
    best_val_acc = -1.0
    epochs_no_improve = 0
    patience = 5
    global_epoch = 0

    # Stage 1
    freeze_all_features(model)
    set_classifier_trainable(model, True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_stage1))

    for e in range(1, args.epochs_stage1 + 1):
        global_epoch += 1
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        lr = optimizer.param_groups[0]["lr"]
        logs.append(EpochResult(global_epoch, "stage1", train_loss, train_acc, val_loss, val_acc, lr))
        print(f"epoch={global_epoch} stage=stage1 train_loss={train_loss:.4f} train_accuracy={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_accuracy={val_acc:.4f} learning_rate={lr:.8f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_no_improve += 1
        torch.save(model.state_dict(), output_dir / "last_model.pth")
        scheduler.step()

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Stage 2
    if epochs_no_improve < patience:
        unfreeze_last_two_feature_blocks(model)
        set_classifier_trainable(model, True)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_stage2))

        for e in range(1, args.epochs_stage2 + 1):
            global_epoch += 1
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            lr = optimizer.param_groups[0]["lr"]
            logs.append(EpochResult(global_epoch, "stage2", train_loss, train_acc, val_loss, val_acc, lr))
            print(f"epoch={global_epoch} stage=stage2 train_loss={train_loss:.4f} train_accuracy={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_accuracy={val_acc:.4f} learning_rate={lr:.8f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            else:
                epochs_no_improve += 1
            torch.save(model.state_dict(), output_dir / "last_model.pth")
            scheduler.step()

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    with (output_dir / "class_names_used.json").open("w", encoding="utf-8") as f:
        json.dump(used_classes, f, ensure_ascii=False, indent=2)

    save_training_log(logs, output_dir / "training_log.csv")

    # Confusion matrix/report based on best model
    best_model = models.mobilenet_v3_small(weights=None)
    in_features = best_model.classifier[-1].in_features
    best_model.classifier[-1] = nn.Linear(in_features, len(used_classes))
    best_model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    best_model.to(device)

    y_true, y_pred = evaluate_and_collect(best_model, val_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(used_classes))))
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + used_classes)
        for i, row in enumerate(cm):
            writer.writerow([used_classes[i]] + row.tolist())

    report = classification_report(y_true, y_pred, labels=list(range(len(used_classes))),
                                   target_names=used_classes, output_dict=True, zero_division=0)
    with (output_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1-score", "support"])
        for class_name in used_classes:
            r = report.get(class_name, {})
            writer.writerow([
                class_name,
                r.get("precision", 0.0),
                r.get("recall", 0.0),
                r.get("f1-score", 0.0),
                r.get("support", 0),
            ])

    model_info = {
        "model_name": "mobilenet_v3_small",
        "image_size": IMAGE_SIZE,
        "used_classes": used_classes,
        "skipped_classes": skipped_classes,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "best_val_accuracy": best_val_acc,
        "epochs_completed": len(logs),
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "model_info.json").open("w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
