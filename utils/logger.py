import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ProTracker",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    file_mode: str = "a",
) -> logging.Logger:
    """
    Set up a logger for ProTracker.

    Args:
        name: Logger name.
        log_file: Optional path to save log file.
        level: Logging level.
        file_mode: File writing mode. "a" means append, "w" means overwrite.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            filename=str(log_file),
            mode=file_mode,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ProTracker") -> logging.Logger:
    """
    Get an existing logger.

    Args:
        name: Logger name.

    Returns:
        Logger object.
    """
    return logging.getLogger(name)


def log_config(logger: logging.Logger, config: dict, prefix: str = "") -> None:
    """
    Print configuration dictionary in a readable form.

    Args:
        logger: Logger object.
        config: Configuration dictionary.
        prefix: Prefix string for nested keys.
    """
    for key, value in config.items():
        current_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            log_config(logger, value, prefix=current_key)
        else:
            logger.info(f"{current_key}: {value}")


class AverageMeter:
    """
    Average meter for recording training or evaluation values.

    This class is commonly used to record losses, metrics, running time,
    and other scalar values.
    """

    def __init__(self, name: str = "meter") -> None:
        """
        Args:
            name: Meter name.
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """
        Reset meter values.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter value.

        Args:
            val: Current value.
            n: Number of samples.
        """
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        """
        Convert meter to string.

        Returns:
            Readable meter string.
        """
        return f"{self.name}: val={self.val:.4f}, avg={self.avg:.4f}"


class ProgressLogger:
    """
    Progress logger for training and testing loops.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total_steps: int,
        print_interval: int = 20,
        prefix: str = "",
    ) -> None:
        """
        Args:
            logger: Logger object.
            total_steps: Total number of steps.
            print_interval: Print interval.
            prefix: Prefix string.
        """
        self.logger = logger
        self.total_steps = total_steps
        self.print_interval = print_interval
        self.prefix = prefix

    def log(
        self,
        step: int,
        meters: Optional[list] = None,
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Print progress.

        Args:
            step: Current step.
            meters: List of AverageMeter objects.
            extra_info: Optional extra information.
        """
        if step % self.print_interval != 0 and step != self.total_steps:
            return

        message = f"{self.prefix}[{step}/{self.total_steps}]"

        if meters is not None:
            meter_text = " | ".join(str(meter) for meter in meters)
            message += f" {meter_text}"

        if extra_info is not None:
            message += f" | {extra_info}"

        self.logger.info(message)