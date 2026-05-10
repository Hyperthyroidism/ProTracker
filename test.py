import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load testing configuration from a yaml file.

    Args:
        config_path: Path to yaml configuration file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def build_device(config: Dict[str, Any]) -> torch.device:
    """
    Build inference device from config.

    Args:
        config: Configuration dictionary.

    Returns:
        torch.device object.
    """
    device_name = config.get("project", {}).get("device", "cuda")

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        device_name = "cpu"

    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test ProTracker on Vessel-MOT dataset."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/test/ProTracker.yaml",
        help="Path to the testing configuration file."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to ProTracker checkpoint."
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory of testing videos. If provided, it will override the path in config."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save testing results."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, such as cuda or cpu."
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after testing."
    )

    return parser.parse_args()


def update_config_by_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update config according to command line arguments.

    Args:
        config: Original configuration dictionary.
        args: Command line arguments.

    Returns:
        Updated configuration dictionary.
    """
    if args.device is not None:
        config["project"]["device"] = args.device

        if "detector" in config:
            config["detector"]["device"] = args.device

        if "sam2" in config:
            config["sam2"]["device"] = args.device

    if args.weights is not None:
        config["graph_prompt_optimizer"]["checkpoint_path"] = args.weights

    if args.video_dir is not None:
        config["dataset"]["video_dir"] = args.video_dir

    if args.output is not None:
        config["tracking"]["result_dir"] = args.output
        config["evaluation"]["pred_dir"] = args.output

    if args.eval:
        config["evaluation"]["enabled"] = True

    return config


def collect_test_videos(config: Dict[str, Any]) -> List[Path]:
    """
    Collect testing videos from dataset directory.

    Args:
        config: Configuration dictionary.

    Returns:
        List of video paths.
    """
    video_dir = Path(config.get("dataset", {}).get("video_dir", "datasets/Vessel-MOT/videos"))

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(video_dir.rglob(f"*{ext}"))

    video_paths = sorted(video_paths)

    if len(video_paths) == 0:
        raise FileNotFoundError(f"No testing videos found in: {video_dir}")

    return video_paths


def run_evaluation(config: Dict[str, Any]) -> None:
    """
    Run evaluation after testing.

    Args:
        config: Configuration dictionary.
    """
    evaluation_cfg = config.get("evaluation", {})
    enabled = evaluation_cfg.get("enabled", False)

    if not enabled:
        return

    gt_dir = evaluation_cfg.get("gt_dir", "datasets/Vessel-MOT/annotations/test")
    pred_dir = evaluation_cfg.get("pred_dir", "results/test")
    metrics = evaluation_cfg.get("metrics", ["MOTA", "HOTA", "IDF1"])

    print("\nStart evaluation.")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Prediction directory: {pred_dir}")
    print(f"Metrics: {metrics}")

    try:
        from evaluate.eval_mota import evaluate_mota
        from evaluate.eval_hota import evaluate_hota
        from evaluate.eval_idf1 import evaluate_idf1
    except ImportError:
        print(
            "\nEvaluation scripts are not fully implemented yet.\n"
            "Please make sure eval_mota.py, eval_hota.py and eval_idf1.py exist in the evaluate directory.\n"
        )
        return

    if "MOTA" in metrics:
        evaluate_mota(gt_dir=gt_dir, pred_dir=pred_dir)

    if "HOTA" in metrics:
        evaluate_hota(gt_dir=gt_dir, pred_dir=pred_dir)

    if "IDF1" in metrics:
        evaluate_idf1(gt_dir=gt_dir, pred_dir=pred_dir)

    print("Evaluation finished.")


class ProTrackerTester:
    """
    Tester for ProTracker.

    This class provides a unified testing process for the whole ProTracker pipeline.
    It loads testing videos, calls the inference pipeline, saves MOT-format tracking
    results, and optionally performs evaluation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = build_device(config)

        tracking_cfg = config.get("tracking", {})
        self.result_dir = Path(tracking_cfg.get("result_dir", "results/test"))
        self.result_dir.mkdir(parents=True, exist_ok=True)

        try:
            from tracker.inference_pipeline import ProTrackerInferencePipeline
        except ImportError as e:
            print(
                "\nFailed to import ProTrackerInferencePipeline.\n"
                "Please make sure tracker/inference_pipeline.py has been created.\n"
            )
            raise e

        self.pipeline = ProTrackerInferencePipeline(config)

    def test_one_video(self, video_path: Path) -> None:
        """
        Test ProTracker on one video.

        Args:
            video_path: Path to input video.
        """
        video_name = video_path.stem
        output_dir = self.result_dir / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTesting video: {video_path}")
        print(f"Output directory: {output_dir}")

        self.pipeline.run(
            video_path=str(video_path),
            output_dir=str(output_dir)
        )

    def test(self) -> None:
        """
        Run testing on all testing videos.
        """
        video_paths = collect_test_videos(self.config)

        print("Start testing ProTracker.")
        print(f"Device: {self.device}")
        print(f"Number of testing videos: {len(video_paths)}")
        print(f"Result directory: {self.result_dir}")

        for video_path in video_paths:
            self.test_one_video(video_path)

        print("\nTesting finished.")
        print(f"All results saved to: {self.result_dir}")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = update_config_by_args(config, args)

    tester = ProTrackerTester(config)
    tester.test()

    run_evaluation(config)


if __name__ == "__main__":
    main()