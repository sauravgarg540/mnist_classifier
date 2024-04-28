import glob
import os
from typing import Callable, Optional

import cv2
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, img_dir: str, transform: Optional[Callable] = None):
        """
        Initializes the MNISTDataset class.

        Args:
            img_dir: Directory containing the images.
            transform: Optional transform to be applied on the images. Defaults to None.
        """
        self.images = glob.glob(os.path.join(img_dir, "*/*.jpg"), recursive=True)
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            length of the dataset
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Returns the image and label at the given index.

        Args:
            idx: Index of the image to be returned.

        Returns:
            Image and label.
        """
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = img_path.split(os.sep)[-2]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(label)).long()
        return image, label
