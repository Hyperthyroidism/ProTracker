import argparse
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def read_yolo_label(label_path: Path) -> List[List[float]]:
    """
    Read one YOLO-format label file.

    YOLO format:
        class_id x_center y_center width height confidence(optional)

    All coordinates are normalized to [0, 1].

    Args:
        label_path: Path to YOLO label file.

    Returns:
        List of YOLO label records.
    """
    records = []

    if not label_path.exists():
        return records

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            values = [float(x) for x in parts]

            if len(values) == 5:
                values.append(1.0)

            records.append(values[:6])

    return records


def yolo_to_mot_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert normalized YOLO box to MOT xywh box.

    Args:
        x_center: Normalized x center.
        y_center: Normalized y center.
        width: Normalized box width.
        height: Normalized box height.
        image_width: Image width.
        image_height: Image height.

    Returns:
        MOT box in x, y, width, height format.
    """
    box_w = width * image_width
    box_h = height * image_height

    x = x_center * image_width - box_w / 2.0
    y = y_center * image_height - box_h / 2.0

    return x, y, box_w, box_h


def find_image_path(image_dir: Path, stem: str) -> Path:
    """
    Find image path according to file stem.

    Args:
        image_dir: Image directory.
        stem: File stem.

    Returns:
        Image path if found, otherwise an empty Path.
    """
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return Path()


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """
    Get image size.

    Args:
        image_path: Image file path.

    Returns:
        Image width and image height.
    """
    import cv2

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]

    return width, height


def convert_one_label_file(
    label_path: Path,
    image_dir: Path,
    frame_id: int,
    image_width: int = 1920,
    image_height: int = 1080,
    use_image_size: bool = True,
) -> List[List[float]]:
    """
    Convert one YOLO label file to MOT lines.

    Args:
        label_path: YOLO label path.
        image_dir: Image directory.
        frame_id: Frame index in MOT format.
        image_width: Default image width.
        image_height: Default image height.
        use_image_size: Whether to read the real image size.

    Returns:
        MOT-format lines.
    """
    if use_image_size:
        image_path = find_image_path(image_dir, label_path.stem)

        if image_path.exists():
            image_width, image_height = get_image_size(image_path)

    yolo_records = read_yolo_label(label_path)

    mot_lines = []

    for obj_idx, record in enumerate(yolo_records):
        class_id, x_center, y_center, width, height, confidence = record

        x, y, w, h = yolo_to_mot_box(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )

        # YOLO labels usually do not contain identity information.
        # Therefore, a temporary track_id is assigned here.
        track_id = obj_idx + 1

        mot_line = [
            float(frame_id),
            float(track_id),
            float(x),
            float(y),
            float(w),
            float(h),
            float(confidence),
            float(class_id),
            1.0,
        ]

        mot_lines.append(mot_line)

    return mot_lines


def save_mot_lines(mot_lines: List[List[float]], save_path: Path) -> None:
    """
    Save MOT-format lines.

    Args:
        mot_lines: MOT-format result lines.
        save_path: Output txt path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mot_lines = sorted(mot_lines, key=lambda x: (int(x[0]), int(x[1])))

    with open(save_path, "w", encoding="utf-8") as f:
        for line in mot_lines:
            line_str = ",".join(
                [
                    str(int(value)) if idx in [0, 1, 7] else f"{float(value):.6f}"
                    for idx, value in enumerate(line)
                ]
            )
            f.write(line_str + "\n")


def convert_yolo_dir_to_mot(
    label_dir: str,
    image_dir: str,
    output_path: str,
    image_width: int = 1920,
    image_height: int = 1080,
    use_image_size: bool = True,
) -> None:
    """
    Convert a directory of YOLO labels to one MOT result file.

    Args:
        label_dir: YOLO label directory.
        image_dir: Image directory.
        output_path: Output MOT txt path.
        image_width: Default image width.
        image_height: Default image height.
        use_image_size: Whether to read the real image size.
    """
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = sorted(label_dir.glob("*.txt"))

    if len(label_paths) == 0:
        raise FileNotFoundError(f"No YOLO label files found in: {label_dir}")

    all_mot_lines = []

    for frame_idx, label_path in enumerate(label_paths, start=1):
        mot_lines = convert_one_label_file(
            label_path=label_path,
            image_dir=image_dir,
            frame_id=frame_idx,
            image_width=image_width,
            image_height=image_height,
            use_image_size=use_image_size,
        )

        all_mot_lines.extend(mot_lines)

    save_mot_lines(all_mot_lines, output_path)

    print(f"Converted {len(label_paths)} YOLO label files.")
    print(f"MOT result saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert YOLO-format labels to MOT-format annotations."
    )

    parser.add_argument(
        "--label-dir",
        type=str,
        required=True,
        help="Directory containing YOLO label txt files."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing image frames."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MOT-format txt file."
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="Default image width if image size cannot be read."
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
        help="Default image height if image size cannot be read."
    )

    parser.add_argument(
        "--no-image-size",
        action="store_true",
        help="Do not read real image size. Use default width and height."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_yolo_dir_to_mot(
        label_dir=args.label_dir,
        image_dir=args.image_dir,
        output_path=args.output,
        image_width=args.image_width,
        image_height=args.image_height,
        use_image_size=not args.no_image_size,
    )


if __name__ == "__main__":
    main()