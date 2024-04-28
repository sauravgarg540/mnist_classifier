import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from config import ASSETS_DIR
from src.utils.utils import logging


def plot_results(images: np.ndarray, pred: np.ndarray, true: np.ndarray) -> None:
    """
    Plot images with their predicted and true labels and stores the results to src/assets folder.

    Args:
        images: Array of images.
        pred: Predicted labels.
        true: True labels.

    Returns:
        None
    """
    # change figure size
    plt.rcParams["figure.figsize"] = (20, 10)

    f, ax = plt.subplots(4, 8)
    count = 0
    for i in range(4):
        for j in range(8):
            ax[i, j].imshow(images[count, 0, :, :], cmap="gray")
            ax[i, j].set_title(f"Predicted:{pred[count]}, Actual:{true[count]}")
            ax[i, j].axis('off')
            count += 1
    plt.savefig(os.path.join(ASSETS_DIR, "predictions.png"))


def plot_confusion_matrix(conf_mat: np.ndarray) -> None:
    """
    Plots the confusion matrix.

    Args:
        conf_mat: Confusion matrix to be plotted.

    Returns:
        None
    """

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.savefig(os.path.join(ASSETS_DIR, 'confusion_matrix.png'))
    logging.info("Confusion matrix saved at assets/confusion_matrix.png")
