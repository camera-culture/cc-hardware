import numpy as np
import torch
from torch import nn

from cc_hardware.utils.constants import TORCH_DEVICE


class DeepLocation8(nn.Module):
    """
    DeepLocation8 model: 2-layer convolutional network designed for 8x8 histogram input
    """

    def __init__(self, height: int, width: int, num_bins: int, out_dims: int = 2):
        super().__init__()

        self.height = height
        self.width = width
        self.num_bins = num_bins

        # in: (n, self.height, self.width, 16)
        self.conv_channels = 4
        self.conv_channels2 = 8
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=(3, 3, 7),
            padding=(1, 1, 3),
        )
        # (n, 4, self.height, self.width, 16)
        self.batchnorm3d = nn.BatchNorm3d(self.conv_channels)
        self.batchnorm3d2 = nn.BatchNorm3d(self.conv_channels2)
        # reshape to (n, 4, self.height x self.width, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 4, self.height, self.width, 8)
        self.conv3d2 = nn.Conv3d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels2,
            kernel_size=(3, 3, 5),
            padding=(1, 1, 2),
        )
        # (n, 8, self.height, self.width, 8)
        # reshape to (n, 8, self.height x self.width, 8)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 8, self.height, self.width, 4)

        self.fc1 = nn.Linear(self.conv_channels2 * self.height * self.width * 4, 128)

        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, out_dims)  # 2 output dimensions (x, y)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.conv3d(x.unsqueeze(1)))
        x = self.batchnorm3d(x)
        x = torch.reshape(
            x,
            (x.shape[0], self.conv_channels * self.height * self.width, self.num_bins),
        )
        x = self.pool1(x)
        x = torch.reshape(
            x, (x.shape[0], self.conv_channels, self.height, self.width, -1)
        )
        x = self.relu(self.conv3d2(x))
        x = self.batchnorm3d2(x)
        x = torch.reshape(
            x, (x.shape[0], self.conv_channels2 * self.height * self.width, -1)
        )
        x = self.pool2(x)
        x = torch.reshape(
            x, (x.shape[0], self.conv_channels2, self.height, self.width, -1)
        )

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x

    def evaluate(self, x: torch.Tensor | np.ndarray | list) -> np.ndarray:
        """
        Evaluate the model on input data x.
        Args:
            x (torch.Tensor): Input tensor of shape (n, height, width, num_bins).
        Returns:
            torch.Tensor: Output tensor of shape (n, out_dims).
        """
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x, dtype=torch.float32, device=TORCH_DEVICE)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a numpy array or a torch tensor."
                f" Got {type(x)} instead."
            )

        if x.ndim == 3:
            x.unsqueeze_(0)

        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy()
