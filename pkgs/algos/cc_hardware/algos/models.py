"""Neural network model architectures and utils"""

import torch
import torch.nn as nn


class DeepLocation8(nn.Module):
    """
    DeepLocation8 model: 2-layer convolutional network designed for 8x8 histogram input
    """
    def __init__(self, height=8, width=8, num_bins=16):
        super(DeepLocation8, self).__init__()

        self.height=height
        self.width=width
        self.num_bins=16

        # in: (n, self.height, self.width, 16)
        self.conv_channels = 4
        self.conv_channels2 = 8
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=self.conv_channels, kernel_size=(3, 3, 7), padding=(1, 1, 3))
        # (n, 4, self.height, self.width, 16)
        self.batchnorm3d = nn.BatchNorm3d(self.conv_channels)
        self.batchnorm3d2 = nn.BatchNorm3d(self.conv_channels2)
        # reshape to (n, 4, self.height x self.width, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 4, self.height, self.width, 8)
        self.conv3d2 = nn.Conv3d(in_channels=self.conv_channels, out_channels=self.conv_channels2, kernel_size=(3, 3, 5), padding=(1, 1, 2))
        # (n, 8, self.height, self.width, 8)
        # reshape to (n, 8, self.height x self.width, 8)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 8, self.height, self.width, 4)

        self.fc1 = nn.Linear(self.conv_channels2 * self.height * self.width * 4, 128)

        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)  # 2 output dimensions (x, y)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.relu(self.conv3d(x.unsqueeze(1)))
        x = self.batchnorm3d(x)
        x = torch.reshape(x, (x.shape[0], self.conv_channels * self.height * self.width, self.num_bins))
        x = self.pool1(x)
        x = torch.reshape(x, (x.shape[0], self.conv_channels, self.height, self.width, -1))
        x = self.relu(self.conv3d2(x))
        x = self.batchnorm3d2(x)
        x = torch.reshape(x, (x.shape[0], self.conv_channels2 * self.height * self.width, -1))
        x = self.pool2(x)
        x = torch.reshape(x, (x.shape[0], self.conv_channels2, self.height, self.width, -1))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x
    


def initialize_weights(m):
    """
    Initialize the weights of the model using Kaiming uniform initialization.
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)