import gzip
import os
from typing import Literal, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import DATA_ROOT_DIR, IMAGE_SIZE, UNPROCESSED_DATA_ROOT_DIR
from src.utils.utils import logging


def load_mnist(path: str, kind: Union[Literal["train", "t10k"]] = 'train'):
    """
    Load MNIST data from `path`.
    Args:
        path: Path to the unprocessed MNIST dataset.
        kind: Kind of dataset to load. Defaults to 'train'.

    Returns:
        images: images in the dataset
        labels: labels in the dataset
    """
    assert kind in ['train', 't10k'], "Kind should be either 'train' or 't10k'"
    logging.info(f"Loading {kind} dataset...")
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def save_mnist(path: str, images: np.ndarray, labels: np.ndarray) -> None:
    """
    Save MNIST data to the given path.
    Args:
        path: Path to save the dataset.
        images: images in the dataset
        labels: labels in the dataset

    Returns:
        None

    """
    logging.info(f"Saving imagenet converted images to {path}...")
    for i, (im, l) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        parent_path = os.path.join(path, str(l))
        os.makedirs(parent_path, exist_ok=True)

        dest = os.path.join(parent_path, f'{i}.jpg')
        im = im.reshape(*IMAGE_SIZE)
        im = Image.fromarray(im, mode='L')
        im.save(dest)


def mnist_to_imagenet_format():
    """
    Convert MNIST dataset to imagenet format.
    Returns:
        None
    """
    logging.info("Converting MNIST dataset to imagenet format...")
    # convert to imagenet image format
    # images, labels = load_mnist(UNPROCESSED_DATA_ROOT_DIR, 'train')
    # save_mnist(os.path.join(DATA_ROOT_DIR, 'train'), images, labels)

    # test
    images, labels = load_mnist(UNPROCESSED_DATA_ROOT_DIR, 't10k')
    save_mnist(os.path.join(DATA_ROOT_DIR, 'test'), images, labels)


if __name__ == '__main__':
    mnist_to_imagenet_format()
