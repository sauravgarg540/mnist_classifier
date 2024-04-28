import torch
import torch.nn as nn

from config import NUM_CLASSES


class MNISTModel(nn.Module):
    """
    Neural network model for image classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        relu_1 (nn.ReLU): ReLU activation function.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool1 (nn.MaxPool2d): Max pooling layer.
        relu_2 (nn.ReLU): ReLU activation function.
        flatten (nn.Flatten): Flatten layer.
        fc1 (nn.Linear): First fully connected layer.
        tan_1 (nn.Tanh): Hyperbolic tangent activation function.
        fc2 (nn.Linear): Final fully connected layer.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the network.
    """

    def __init__(self):
        """
        Initializes the Model class.
        """
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu_2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12544, 128)
        self.tan_1 = nn.Tanh()
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output from the network. Softmax is applied in the loss function.
        """
        x = self.conv1(x)
        x = self.relu_1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.relu_2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tan_1(x)
        x = self.fc2(x)
        return x
