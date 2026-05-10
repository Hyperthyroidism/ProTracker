import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """
    Load yaml configuration file.

    Args:
        config_path: Path to the yaml config file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def update_config_by_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Update configuration according to command line arguments.

    Args:
        config: Original configuration dictionary.
        args: Command line arguments.

    Returns:
        Updated configuration dictionary.
    """
    if "input" not in config:
        config["input"] = {}

    if "tracking" not in config:
        config["tracking"] = {}

    if "visualization" not in config:
        config["visualization"] = {}

    config["input"]["mode"] = "video"
    config["input"]["video_path"] = args.video

    config["tracking"]["result_dir"] = args.output
    config["visualization"]["output_video"] = str(Path(args.output) / "demo_vis.mp4")

    if args.device is not None:
        config["project"]["device"] = args.device

        if "detector" in config:
            config["detector"]["device"] = args.device

        if "sam2" in config:
            config["sam2"]["device"] = args.device

    if args.yolo_weight is not None:
        config["detector"]["model_path"] = args.yolo_weight

    if args.sam2_weight is not None:
        config["sam2"]["checkpoint_path"] = args.sam2_weight

    if args.protracker_weight is not None:
        config["graph_prompt_optimizer"]["checkpoint_path"] = args.protracker_weight

    return config


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run ProTracker on a single UAV video."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/test/ProTracker.yaml",
        help="Path to the testing configuration file."
    )

    parser.add_argument(
        "--video",
        type=str,
        default="assets/demo.mp4",
        help="Path to the input video."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/demo",
        help="Directory to save tracking results."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device used for inference, such as cuda or cpu."
    )

    parser.add_argument(
        "--yolo-weight",
        type=str,
        default=None,
        help="Path to YOLO11 vessel detector weight."
    )

    parser.add_argument(
        "--sam2-weight",
        type=str,
        default=None,
        help="Path to SAM2 checkpoint."
    )

    parser.add_argument(
        "--protracker-weight",
        type=str,
        default=None,
        help="Path to ProTracker graph prompt optimizer checkpoint."
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for demo inference.

    This script runs the complete ProTracker inference pipeline:

        input video
            -> YOLO11 vessel detection
            -> graph-based cascaded prompt optimization
            -> target-aware prompt refinement
            -> SAM2-based vessel segmentation and tracking
            -> MOT-format result and visualization output
    """
    args = parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config = update_config_by_args(config, args)

    try:
        from tracker.inference_pipeline import ProTrackerInferencePipeline
    except ImportError as e:
        print(
            "\nFailed to import ProTrackerInferencePipeline.\n"
            "Please make sure that tracker/inference_pipeline.py exists "
            "and the project root is correctly added to PYTHONPATH.\n"
        )
        raise e

    pipeline = ProTrackerInferencePipeline(config)
    pipeline.run(
        video_path=str(video_path),
        output_dir=str(output_dir)
    )

    print("\nProTracker inference finished.")
    print(f"Input video: {video_path}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()