import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str:
    kwargs = dict(cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], **kwargs).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def setup_logging(logger: logging.Logger, log_path: Path | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
