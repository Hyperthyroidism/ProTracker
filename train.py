import argparse
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from a yaml file.

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


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_device(config: Dict[str, Any]) -> torch.device:
    """
    Build training device from config.

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
        description="Train the graph-based cascaded prompt optimizer of ProTracker."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/ProTracker.yaml",
        help="Path to the training configuration file."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, such as cuda or cpu."
    )

    return parser.parse_args()


class ProTrackerTrainer:
    """
    Trainer for the graph-based cascaded prompt optimizer.

    This trainer is designed as the unified training entrance of ProTracker.
    The main training target is the graph-based prompt optimization module,
    which learns to enhance node representations, refine edge features,
    and generate stable vessel prompts for SAM2-based tracking.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

        training_cfg = config.get("training", {})
        checkpoint_cfg = config.get("checkpoint", {})
        logging_cfg = config.get("logging", {})

        self.epochs = training_cfg.get("epochs", 200)
        self.batch_size = training_cfg.get("batch_size", 4)

        self.save_dir = Path(checkpoint_cfg.get("save_dir", "weights/protracker"))
        self.log_dir = Path(logging_cfg.get("log_dir", "logs/train"))

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

    def build_model(self) -> torch.nn.Module:
        """
        Build graph-based prompt optimizer model.

        Returns:
            ProTracker graph prompt optimizer model.
        """
        try:
            from networks.prompt_optimizer import GraphPromptOptimizer
        except ImportError as e:
            print(
                "\nFailed to import GraphPromptOptimizer.\n"
                "Please make sure networks/prompt_optimizer.py has been created.\n"
            )
            raise e

        model = GraphPromptOptimizer(self.config)
        model = model.to(self.device)

        return model

    def build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build optimizer.

        Returns:
            PyTorch optimizer.
        """
        training_cfg = self.config.get("training", {})
        optimizer_cfg = training_cfg.get("optimizer", {})

        optimizer_name = optimizer_cfg.get("name", "Adam")
        learning_rate = optimizer_cfg.get("learning_rate", 0.0003)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0001)

        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint for resuming training.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Checkpoint loaded from: {checkpoint_path}")

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this checkpoint is the best one.
        """
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }

        checkpoint_path = self.save_dir / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.save_dir / "protracker.pth"
            torch.save(checkpoint, best_path)

        print(f"Checkpoint saved to: {checkpoint_path}")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train model for one epoch.

        This function is written as a standard training interface.
        The actual graph samples and loss computation will be connected
        with the Vessel-MOT graph dataset and GraphPromptOptimizer later.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of training losses.
        """
        self.model.train()

        # Placeholder loss values.
        # The real graph batch loading and loss computation will be implemented
        # together with Vessel-MOT graph data preparation.
        loss_dict = {
            "total_loss": 0.0,
            "edge_loss": 0.0,
            "node_loss": 0.0,
            "prompt_loss": 0.0
        }

        print(
            f"[Epoch {epoch:03d}] "
            f"total_loss: {loss_dict['total_loss']:.4f}, "
            f"edge_loss: {loss_dict['edge_loss']:.4f}, "
            f"node_loss: {loss_dict['node_loss']:.4f}, "
            f"prompt_loss: {loss_dict['prompt_loss']:.4f}"
        )

        return loss_dict

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        metric_dict = {
            "MOTA": 0.0,
            "HOTA": 0.0,
            "IDF1": 0.0
        }

        print(
            f"[Validation {epoch:03d}] "
            f"MOTA: {metric_dict['MOTA']:.4f}, "
            f"HOTA: {metric_dict['HOTA']:.4f}, "
            f"IDF1: {metric_dict['IDF1']:.4f}"
        )

        return metric_dict

    def train(self) -> None:
        """
        Run the complete training process.
        """
        validation_cfg = self.config.get("validation", {})
        checkpoint_cfg = self.config.get("checkpoint", {})

        val_enabled = validation_cfg.get("enabled", True)
        val_interval = validation_cfg.get("interval", 1)

        save_interval = self.config.get("logging", {}).get("save_interval", 10)
        save_best = checkpoint_cfg.get("save_best", True)
        monitor_metric = checkpoint_cfg.get("monitor_metric", "IDF1")

        best_score = -1.0

        print("Start training ProTracker.")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Checkpoint directory: {self.save_dir}")

        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)

            is_best = False

            if val_enabled and epoch % val_interval == 0:
                metrics = self.validate(epoch)
                current_score = metrics.get(monitor_metric, 0.0)

                if current_score > best_score:
                    best_score = current_score
                    is_best = True

            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

        print("Training finished.")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    if args.device is not None:
        config["project"]["device"] = args.device

    if args.resume is not None:
        config["checkpoint"]["resume"] = True
        config["checkpoint"]["resume_path"] = args.resume

    seed = config.get("project", {}).get("seed", 42)
    set_random_seed(seed)

    device = build_device(config)

    trainer = ProTrackerTrainer(config, device)

    checkpoint_cfg = config.get("checkpoint", {})
    if checkpoint_cfg.get("resume", False):
        resume_path = checkpoint_cfg.get("resume_path", None)
        if resume_path is not None:
            trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()