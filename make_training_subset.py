#!/usr/bin/env python3
"""Build balanced dataset_train_ready from dataset_final.

Creates a capped train/val subset per class while preserving the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List

DEFAULT_SOURCE_ROOT = r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset\dataset_final"
DEFAULT_OUTPUT_ROOT = r"C:\Users\zudae\Desktop\диплом1\model3\fashion-dataset\dataset_train_ready"
DEFAULT_MAX_TRAIN_PER_CLASS = 5000
DEFAULT_MAX_VAL_PER_CLASS = 1000
DEFAULT_SEED = 42


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create balanced training subset dataset_train_ready from dataset_final."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(DEFAULT_SOURCE_ROOT),
        help="Path to source dataset_final.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
        help="Path to output dataset_train_ready.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=DEFAULT_MAX_TRAIN_PER_CLASS,
        help="Maximum train files per class.",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=DEFAULT_MAX_VAL_PER_CLASS,
        help="Maximum val files per class.",
    )
    return parser.parse_args()


def load_class_names(source_root: Path) -> List[str]:
    class_names_path = source_root / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class_names.json: {class_names_path}")

    with class_names_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "class_names" in data and isinstance(data["class_names"], list):
            return [str(x) for x in data["class_names"]]
        return [str(k) for k in data.keys()]

    raise ValueError("Unsupported class_names.json format")


def list_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def choose_subset(files: List[Path], max_count: int, rng: random.Random) -> List[Path]:
    if max_count < 0:
        raise ValueError("max_count cannot be negative")
    if len(files) <= max_count:
        return files
    return rng.sample(files, max_count)


def ensure_clean_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)


def copy_files(files: Iterable[Path], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in files:
        dst = out_dir / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            idx = 1
            while dst.exists():
                dst = out_dir / f"{stem}_{idx}{suffix}"
                idx += 1
        shutil.copy2(src, dst)
        copied += 1
    return copied


def write_class_names(class_names: List[str], output_root: Path) -> None:
    with (output_root / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


def write_stats(rows: List[dict], output_root: Path) -> None:
    stats_path = output_root / "dataset_stats.csv"
    fieldnames = [
        "class_name",
        "source_train_count",
        "source_val_count",
        "train_count",
        "val_count",
    ]
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    source_root: Path = args.source_root
    output_root: Path = args.output_root
    max_train_per_class: int = args.max_train_per_class
    max_val_per_class: int = args.max_val_per_class

    class_names = load_class_names(source_root)
    ensure_clean_output(output_root)

    rng = random.Random(DEFAULT_SEED)

    total_copied_train = 0
    total_copied_val = 0
    empty_classes: List[str] = []
    classes_lt_100: List[str] = []
    stats_rows: List[dict] = []

    for class_name in class_names:
        src_train_dir = source_root / "train" / class_name
        src_val_dir = source_root / "val" / class_name

        train_files = list_files(src_train_dir)
        val_files = list_files(src_val_dir)

        source_train_count = len(train_files)
        source_val_count = len(val_files)
        source_total = source_train_count + source_val_count

        if source_total == 0:
            empty_classes.append(class_name)
        if source_total < 100:
            classes_lt_100.append(class_name)

        selected_train = choose_subset(train_files, max_train_per_class, rng)
        selected_val = choose_subset(val_files, max_val_per_class, rng)

        copied_train = copy_files(selected_train, output_root / "train" / class_name)
        copied_val = copy_files(selected_val, output_root / "val" / class_name)

        total_copied_train += copied_train
        total_copied_val += copied_val

        stats_rows.append(
            {
                "class_name": class_name,
                "source_train_count": source_train_count,
                "source_val_count": source_val_count,
                "train_count": copied_train,
                "val_count": copied_val,
            }
        )

    write_class_names(class_names, output_root)
    write_stats(stats_rows, output_root)

    print("\n=== Training Subset Build Summary ===")
    print(f"Classes processed: {len(class_names)}")
    print(f"Total copied train files: {total_copied_train}")
    print(f"Total copied val files: {total_copied_val}")

    print("\nEmpty classes (0 source images):")
    if empty_classes:
        for cls in empty_classes:
            print(f"  - {cls}")
    else:
        print("  None")

    print("\nClasses with fewer than 100 source images:")
    if classes_lt_100:
        for cls in classes_lt_100:
            print(f"  - {cls}")
    else:
        print("  None")


if __name__ == "__main__":
    main()
