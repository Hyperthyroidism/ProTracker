import argparse
import subprocess
import sys
from pathlib import Path


def check_file_exists(file_path: str, file_type: str = "file") -> None:
    """
    Check whether a file exists.

    Args:
        file_path: File path.
        file_type: Description of file type.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_type} not found: {file_path}")


def run_demo(
    video_path: str,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
) -> None:
    """
    Run ProTracker demo.

    Args:
        video_path: Input demo video path.
        config_path: Testing configuration path.
        output_dir: Output directory.
        device: Inference device.
    """
    check_file_exists(video_path, "Demo video")
    check_file_exists(config_path, "Config file")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "run.py",
        "--config",
        config_path,
        "--video",
        video_path,
        "--output",
        str(output_dir),
        "--device",
        device,
    ]

    print("Running command:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print("\nDemo finished.")
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a quick ProTracker demo."
    )

    parser.add_argument(
        "--video",
        type=str,
        default="assets/demo.mp4",
        help="Path to demo video."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/test/ProTracker.yaml",
        help="Path to testing configuration file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/demo",
        help="Directory to save demo results."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device, such as cuda or cpu."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_demo(
        video_path=args.video,
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()