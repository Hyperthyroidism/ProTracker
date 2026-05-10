import argparse
from pathlib import Path
from typing import Dict, List


REQUIRED_DIRS = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
    "videos/train",
    "videos/val",
    "videos/test",
    "annotations/train",
    "annotations/val",
    "annotations/test",
]


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
ANNOTATION_EXTENSIONS = [".txt", ".json", ".csv"]


def create_dataset_dirs(dataset_root: Path) -> None:
    """
    Create required Vessel-MOT dataset directories.

    Args:
        dataset_root: Root directory of Vessel-MOT dataset.
    """
    for sub_dir in REQUIRED_DIRS:
        dir_path = dataset_root / sub_dir
        dir_path.mkdir(parents=True, exist_ok=True)


def count_files(directory: Path, extensions: List[str]) -> int:
    """
    Count files with given extensions.

    Args:
        directory: Directory to search.
        extensions: File extensions.

    Returns:
        Number of matched files.
    """
    if not directory.exists():
        return 0

    count = 0

    for ext in extensions:
        count += len(list(directory.rglob(f"*{ext}")))

    return count


def summarize_split(dataset_root: Path, split: str) -> Dict[str, int]:
    """
    Summarize one dataset split.

    Args:
        dataset_root: Root directory of Vessel-MOT dataset.
        split: Dataset split, such as train, val, or test.

    Returns:
        Summary dictionary.
    """
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    video_dir = dataset_root / "videos" / split
    annotation_dir = dataset_root / "annotations" / split

    summary = {
        "images": count_files(image_dir, IMAGE_EXTENSIONS),
        "labels": count_files(label_dir, ANNOTATION_EXTENSIONS),
        "videos": count_files(video_dir, VIDEO_EXTENSIONS),
        "annotations": count_files(annotation_dir, ANNOTATION_EXTENSIONS),
    }

    return summary


def print_summary(dataset_root: Path) -> None:
    """
    Print dataset summary.

    Args:
        dataset_root: Root directory of Vessel-MOT dataset.
    """
    print("\nVessel-MOT Dataset Summary")
    print(f"Dataset root: {dataset_root}")
    print("-" * 60)

    for split in ["train", "val", "test"]:
        summary = summarize_split(dataset_root, split)

        print(f"[{split}]")
        print(f"  images:      {summary['images']}")
        print(f"  labels:      {summary['labels']}")
        print(f"  videos:      {summary['videos']}")
        print(f"  annotations: {summary['annotations']}")
        print("-" * 60)


def check_dataset_dirs(dataset_root: Path) -> None:
    """
    Check whether required directories exist.

    Args:
        dataset_root: Root directory of Vessel-MOT dataset.
    """
    missing_dirs = []

    for sub_dir in REQUIRED_DIRS:
        dir_path = dataset_root / sub_dir

        if not dir_path.exists():
            missing_dirs.append(dir_path)

    if len(missing_dirs) == 0:
        print("All required dataset directories exist.")
        return

    print("Missing directories:")

    for dir_path in missing_dirs:
        print(f"  {dir_path}")


def write_dataset_readme(dataset_root: Path) -> None:
    """
    Write a simple README file inside Vessel-MOT dataset directory.

    Args:
        dataset_root: Root directory of Vessel-MOT dataset.
    """
    readme_path = dataset_root / "README.md"

    content = """# Vessel-MOT Dataset

This directory stores the Vessel-MOT dataset used by ProTracker.

## Directory Structure

```text
Vessel-MOT/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── videos/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/