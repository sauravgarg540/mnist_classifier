import numpy as np
import torch


class ToTorchFormatTensor(object):

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert the input image to PyTorch tensor format.

        Args:
            image: Input image in numpy format.

        Returns:
            Converted image in PyTorch tensor format.
        """
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).contiguous()
        return image.float()


class Normalize(object):

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input image by diving by 255.0.

        Args:
            image: Input image.

        Returns:
            Normalized image.
        """
        image = image / 255.0
        return image
