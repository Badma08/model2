#!/usr/bin/env python3
"""Add DeepFashion2 crops into an existing dataset_final structure.

This script reads DeepFashion2 annotations from train/annos and validation/annos,
crops item bounding boxes from corresponding images, maps categories into existing
project classes, and appends the crops to dataset_final/{train,val}/<class_name>.

It also writes:
- dataset_final/deepfashion2_added_stats.csv
- updated dataset_final/dataset_stats.csv with actual file counts
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
from tqdm import tqdm


DEFAULT_DEEPFASHION2_ROOT = r"C:\Users\zudae\Desktop\диплом1\DeepFashion2"
DEFAULT_DATASET_FINAL_ROOT = (
    r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset\dataset_final"
)

SAFE_CATEGORY_MAPPING: Dict[str, str] = {
    "short sleeve top": "tshirt",
    "long sleeve top": "longsleeve",
    "short sleeve outwear": "jacket",
    "long sleeve outwear": "jacket",
    "vest": "vest",
    "shorts": "shorts",
    "trousers": "trousers",
    "skirt": "skirt",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "vest dress": "dress",
    "sling dress": "dress",
    "sling": "top",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add DeepFashion2 crops into existing dataset_final structure."
    )
    parser.add_argument(
        "--deepfashion2-root",
        type=Path,
        default=Path(DEFAULT_DEEPFASHION2_ROOT),
        help="Path to DeepFashion2 root (contains train/ and validation/).",
    )
    parser.add_argument(
        "--dataset-final-root",
        type=Path,
        default=Path(DEFAULT_DATASET_FINAL_ROOT),
        help="Path to dataset_final root.",
    )
    return parser.parse_args()


def load_class_names(dataset_final_root: Path) -> set[str]:
    class_names_path = dataset_final_root / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f"class_names.json not found: {class_names_path}")

    with class_names_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        # Compatible with either {"class_names": [...]} or {"name": ...} style maps.
        if "class_names" in data and isinstance(data["class_names"], list):
            return {str(x) for x in data["class_names"]}
        return {str(k) for k in data.keys()}

    raise ValueError("Unsupported class_names.json format.")


def clamp_bbox(
    bbox: List[float], width: int, height: int
) -> Tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None

    # Clamp to image boundaries.
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def iter_annotation_files(annos_dir: Path) -> Iterable[Path]:
    return sorted(annos_dir.glob("*.json"))


def process_split(
    split_name: str,
    deepfashion2_split_dir: Path,
    output_split_dir: Path,
    allowed_classes: set[str],
    added_counter: Counter,
    global_stats: Counter,
) -> int:
    annos_dir = deepfashion2_split_dir / "annos"
    image_dir = deepfashion2_split_dir / "image"

    if not annos_dir.exists():
        raise FileNotFoundError(f"Annotation directory missing: {annos_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory missing: {image_dir}")

    ann_files = list(iter_annotation_files(annos_dir))
    processed_json = 0

    for ann_path in tqdm(ann_files, desc=f"Processing {split_name} annotations", unit="json"):
        processed_json += 1
        image_id = ann_path.stem
        image_path = image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            global_stats["missing_image"] += 1
            continue

        try:
            with ann_path.open("r", encoding="utf-8") as f:
                ann_data = json.load(f)
        except Exception:
            global_stats["invalid_json"] += 1
            continue

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                for key, item_data in ann_data.items():
                    if not key.startswith("item") or not isinstance(item_data, dict):
                        continue

                    category_name = item_data.get("category_name")
                    bbox = item_data.get("bounding_box")

                    mapped_class = SAFE_CATEGORY_MAPPING.get(str(category_name).strip().lower())
                    if mapped_class is None:
                        global_stats["unknown_category"] += 1
                        continue

                    if mapped_class not in allowed_classes:
                        global_stats["class_not_allowed"] += 1
                        continue

                    valid_bbox = clamp_bbox(bbox, width, height) if isinstance(bbox, list) else None
                    if valid_bbox is None:
                        global_stats["invalid_bbox"] += 1
                        continue

                    x1, y1, x2, y2 = valid_bbox
                    crop_w, crop_h = x2 - x1, y2 - y1
                    if crop_w < 32 or crop_h < 32:
                        global_stats["too_small_crop"] += 1
                        continue

                    crop = img.crop((x1, y1, x2, y2)).convert("RGB")

                    dst_dir = output_split_dir / mapped_class
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    out_name = f"df2_{split_name}_{image_id}_{key}.jpg"
                    out_path = dst_dir / out_name

                    # Ensure uniqueness if file already exists.
                    suffix = 1
                    while out_path.exists():
                        out_name = f"df2_{split_name}_{image_id}_{key}_{suffix}.jpg"
                        out_path = dst_dir / out_name
                        suffix += 1

                    crop.save(out_path, format="JPEG", quality=95)
                    added_counter[mapped_class] += 1
                    global_stats["added_total"] += 1

        except Exception:
            global_stats["image_read_error"] += 1
            continue

    return processed_json


def write_added_stats(dataset_final_root: Path, added_counter: Counter) -> None:
    out_csv = dataset_final_root / "deepfashion2_added_stats.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "added_count"])
        writer.writeheader()
        for class_name in sorted(added_counter.keys()):
            count = added_counter[class_name]
            if count > 0:
                # split="all" because crops were aggregated from train+validation additions.
                writer.writerow(
                    {"split": "all", "class_name": class_name, "added_count": count}
                )


def recalc_and_update_dataset_stats(dataset_final_root: Path, allowed_classes: set[str]) -> None:
    stats_path = dataset_final_root / "dataset_stats.csv"

    train_root = dataset_final_root / "train"
    val_root = dataset_final_root / "val"

    actual_counts = {}
    for class_name in sorted(allowed_classes):
        train_count = (
            len([p for p in (train_root / class_name).glob("*") if p.is_file()])
            if (train_root / class_name).exists()
            else 0
        )
        val_count = (
            len([p for p in (val_root / class_name).glob("*") if p.is_file()])
            if (val_root / class_name).exists()
            else 0
        )
        actual_counts[class_name] = {
            "class_name": class_name,
            "train_count": train_count,
            "val_count": val_count,
            "total_available": train_count + val_count,
        }

    # If old file exists and has extra columns, keep them where possible.
    old_rows = {}
    old_fieldnames: List[str] = []
    if stats_path.exists():
        with stats_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            for row in reader:
                cls = row.get("class_name")
                if cls:
                    old_rows[cls] = row

    required_cols = ["class_name", "train_count", "val_count", "total_available"]
    fieldnames = required_cols[:]
    for col in old_fieldnames:
        if col not in fieldnames:
            fieldnames.append(col)

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in sorted(actual_counts.keys()):
            base = {k: "" for k in fieldnames}
            if class_name in old_rows:
                base.update(old_rows[class_name])

            base.update(
                {
                    "class_name": class_name,
                    "train_count": actual_counts[class_name]["train_count"],
                    "val_count": actual_counts[class_name]["val_count"],
                    "total_available": actual_counts[class_name]["total_available"],
                }
            )
            writer.writerow(base)


def main() -> None:
    args = parse_args()

    deepfashion2_root: Path = args.deepfashion2_root
    dataset_final_root: Path = args.dataset_final_root

    allowed_classes = load_class_names(dataset_final_root)

    added_counter_train: Counter = Counter()
    added_counter_val: Counter = Counter()
    global_stats: Counter = Counter()

    processed_train = process_split(
        split_name="train",
        deepfashion2_split_dir=deepfashion2_root / "train",
        output_split_dir=dataset_final_root / "train",
        allowed_classes=allowed_classes,
        added_counter=added_counter_train,
        global_stats=global_stats,
    )

    processed_val = process_split(
        split_name="val",
        deepfashion2_split_dir=deepfashion2_root / "validation",
        output_split_dir=dataset_final_root / "val",
        allowed_classes=allowed_classes,
        added_counter=added_counter_val,
        global_stats=global_stats,
    )

    added_counter_all = added_counter_train + added_counter_val

    write_added_stats(dataset_final_root, added_counter_all)
    recalc_and_update_dataset_stats(dataset_final_root, allowed_classes)

    print("\n=== DeepFashion2 Add Summary ===")
    print(f"Train JSON processed: {processed_train}")
    print(f"Validation JSON processed: {processed_val}")
    print(f"Crops added total: {global_stats['added_total']}")
    print(f"Skipped unknown category_name: {global_stats['unknown_category']}")
    print(f"Skipped invalid bbox: {global_stats['invalid_bbox']}")
    print(f"Skipped missing image: {global_stats['missing_image']}")
    print(f"Skipped class not in class_names.json: {global_stats['class_not_allowed']}")
    print(f"Skipped too small crop (<32x32): {global_stats['too_small_crop']}")
    print(f"Skipped invalid JSON: {global_stats['invalid_json']}")
    print(f"Skipped image read errors: {global_stats['image_read_error']}")

    print("\nAdded crops by class:")
    if added_counter_all:
        for class_name in sorted(added_counter_all.keys()):
            print(f"  {class_name}: {added_counter_all[class_name]}")
    else:
        print("  No crops were added.")


if __name__ == "__main__":
    main()
