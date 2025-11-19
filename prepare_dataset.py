#!/usr/bin/env python3
"""
Merge dataset1/2/3 exported from Roboflow, drop unwanted classes,
and generate a unified Ultralytics YOLO data.yaml file.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_NAMES = ["dataset1", "dataset2", "dataset3"]
DROP_CLASS_NAMES = {"laptop", "notebook"}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")

COMBINED_ROOT = PROJECT_ROOT / "datasets" / "combined"
DATA_YAML_PATH = PROJECT_ROOT / "data.yaml"


@dataclass
class DatasetInfo:
    name: str
    path: Path
    classes: List[str]


def load_dataset_info(name: str) -> DatasetInfo:
    dataset_path = PROJECT_ROOT / name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Missing data.yaml for {name}: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_names = data.get("names")
    if raw_names is None:
        raise ValueError(f"'names' field is missing in {data_yaml_path}")

    if isinstance(raw_names, dict):
        names = [raw_names[key] for key in sorted(raw_names.keys(), key=int)]
    elif isinstance(raw_names, list):
        names = raw_names
    else:
        raise TypeError(f"Unsupported 'names' type in {data_yaml_path}: {type(raw_names)}")

    return DatasetInfo(name=name, path=dataset_path, classes=names)


def build_final_class_list(datasets: List[DatasetInfo]) -> List[str]:
    ordered_classes: List[str] = []
    for dataset in datasets:
        for cls_name in dataset.classes:
            if cls_name in DROP_CLASS_NAMES:
                continue
            if cls_name not in ordered_classes:
                ordered_classes.append(cls_name)
    if not ordered_classes:
        raise ValueError("No classes remain after filtering. Check DROP_CLASS_NAMES.")
    return ordered_classes


def prepare_output_dirs() -> None:
    if COMBINED_ROOT.exists():
        print(f"Removing previous merged dataset: {COMBINED_ROOT}")
        shutil.rmtree(COMBINED_ROOT)
    for split in SPLITS:
        (COMBINED_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (COMBINED_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def convert_label_lines(
    lines: List[str],
    source_classes: List[str],
    class_lookup: Dict[str, int],
) -> tuple[List[str], int]:
    converted: List[str] = []
    dropped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            cls_idx = int(parts[0])
        except ValueError:
            dropped += 1
            continue
        if cls_idx < 0 or cls_idx >= len(source_classes):
            dropped += 1
            continue
        cls_name = source_classes[cls_idx]
        if cls_name in DROP_CLASS_NAMES:
            dropped += 1
            continue
        new_idx = class_lookup.get(cls_name)
        if new_idx is None:
            dropped += 1
            continue
        parts[0] = str(new_idx)
        converted.append(" ".join(parts))
    return converted, dropped


def copy_split(
    dataset: DatasetInfo,
    split: str,
    class_lookup: Dict[str, int],
) -> Dict[str, int]:
    img_src_dir = dataset.path / split / "images"
    lbl_src_dir = dataset.path / split / "labels"
    if not img_src_dir.exists():
        print(f"Warning: {dataset.name}/{split}/images not found. Skipping.")
        return {"images": 0, "kept_boxes": 0, "dropped_boxes": 0}

    dest_img_dir = COMBINED_ROOT / split / "images"
    dest_lbl_dir = COMBINED_ROOT / split / "labels"

    images_copied = 0
    boxes_kept = 0
    boxes_dropped = 0

    for img_path in sorted(img_src_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        label_path = lbl_src_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        else:
            raw_lines = []

        converted_lines, dropped = convert_label_lines(raw_lines, dataset.classes, class_lookup)
        boxes_kept += len(converted_lines)
        boxes_dropped += dropped

        dest_stem = f"{dataset.name}_{img_path.stem}"
        dest_img_path = dest_img_dir / f"{dest_stem}{img_path.suffix.lower()}"
        dest_lbl_path = dest_lbl_dir / f"{dest_stem}.txt"

        shutil.copy2(img_path, dest_img_path)
        with open(dest_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(converted_lines))

        images_copied += 1

    return {"images": images_copied, "kept_boxes": boxes_kept, "dropped_boxes": boxes_dropped}


def write_data_yaml(final_classes: List[str]) -> None:
    data = {
        "path": str(COMBINED_ROOT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(final_classes),
        "names": final_classes,
    }
    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"\nUpdated data.yaml: {DATA_YAML_PATH}")


def main() -> None:
    print("Starting dataset merge...")
    datasets = [load_dataset_info(name) for name in DATASET_NAMES]
    final_classes = build_final_class_list(datasets)
    class_lookup = {name: idx for idx, name in enumerate(final_classes)}

    print(f"Keeping {len(final_classes)} classes: {final_classes}")
    if DROP_CLASS_NAMES:
        print(f"Dropping classes: {sorted(DROP_CLASS_NAMES)}")

    prepare_output_dirs()

    total_summary = {split: {"images": 0, "kept_boxes": 0, "dropped_boxes": 0} for split in SPLITS}
    for dataset in datasets:
        print(f"\nProcessing {dataset.name}...")
        for split in SPLITS:
            stats = copy_split(dataset, split, class_lookup)
            total_summary[split]["images"] += stats["images"]
            total_summary[split]["kept_boxes"] += stats["kept_boxes"]
            total_summary[split]["dropped_boxes"] += stats["dropped_boxes"]
            print(
                f"   - {split:<5}: {stats['images']:4d} images, "
                f"{stats['kept_boxes']:4d} boxes (dropped {stats['dropped_boxes']})"
            )

    write_data_yaml(final_classes)

    print("\nMerge summary")
    for split in SPLITS:
        summary = total_summary[split]
        print(
            f" - {split:<5}: {summary['images']:4d} images, "
            f"{summary['kept_boxes']:5d} boxes (dropped {summary['dropped_boxes']})"
        )

    print("\nNext step:")
    print("    python train.py")


if __name__ == "__main__":
    main()

