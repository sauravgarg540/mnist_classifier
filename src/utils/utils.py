import logging
import random
from typing import List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger to log the output to both console and file.
    Args:
        name: name of the logger
        level: logging level
    """

    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')

    # Create a handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Configure the logger (usually done at the start of the program)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class ConfusionMatrix:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def compute(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_per_class_accuracy(self):
        conf_mat = self.compute()
        return conf_mat.diagonal() / conf_mat.sum(axis=1)


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric over time.

    Attributes:
        history (List[float]): A list to store the history of values.
        val (float): The current value of the metric.
        avg (float): The average value of the metric.
        sum (float): The sum of all values of the metric.
        count (int): The number of times the metric has been updated.

    Methods:
        reset(): Resets the meter to its initial state.
        update(val: float, n: int = 1): Updates the meter with a new value.
    """

    def __init__(self) -> None:
        """
        Initializes a new AverageMeter object.
        """
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        """
        Resets the meter to its initial state.

        Returns:
            None
        """
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the meter with a new value.

        Args:
            val: The new value to be added to the meter.
            n: The number of times the value is counted. Defaults to 1.

        Returns
            None
        """
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device() -> str:
    """
    Get device (cuda or cpu) based on availability
    Returns:
        device name 'cuda' or 'cpu'
    """
    import torch

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def fix_seed(seed: int = 0) -> None:
    """
    Fix seed for reproducibility and benchmarking
    Args:
        seed: any random seed value

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    setup_cudnn()


def setup_cudnn() -> None:
    """
    Setup cudnn for reproducibility
    read more at https://pytorch.org/docs/stable/notes/randomness.html#cudnn

    Returns:
        None
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


logging = setup_logger("MNIST")
