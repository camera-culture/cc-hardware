import torch
from torch import nn
import torch.nn.functional as F


class RegressionModel(nn.Module):
    def __init__(
        self,
        num_bins: int,
        resolution: tuple[int, int],
        output_size: int,
        *,
        min_bin: int | None = None,
        max_bin: int | None = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin = min_bin or 0
        self.max_bin = max_bin or num_bins
        input_size = (self.max_bin - self.min_bin) * resolution[0] * resolution[1]
        self.all_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.merged_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.max_bin - self.min_bin, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self.out_fc = nn.Linear(output_size * 2, output_size)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        x = x[..., self.min_bin : self.max_bin].float()

        # Add noise (only in training mode)
        if self.training:
            x = x + 1e-4 * x.std() * torch.randn_like(x)

        # Normalize
        x = x / x.sum(dim=1, keepdim=True)
        all_x_features = self.all_fc(x)
        x_merged = x.sum(dim=1)
        merged_x_features = self.merged_fc(x_merged)
        return self.out_fc(torch.cat([all_x_features, merged_x_features], dim=1))


class RegressionModel(nn.Module):
    def __init__(
        self,
        num_bins: int,
        resolution: tuple[int, int],
        output_size: int,
        *,
        min_bin: int | None = None,
        max_bin: int | None = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin = min_bin or 0
        self.max_bin = max_bin or num_bins
        self.resolution = resolution

        # Determine input size after slicing
        input_bins = self.max_bin - self.min_bin

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=resolution[0] * resolution[1],
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        conv_output_size = 128 * (
            input_bins // 4
        )  # Two pooling layers reduce size by factor of 4
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # Slice input to specified bins
        x = x[..., self.min_bin : self.max_bin].float()

        # Add noise (only in training mode)
        if self.training:
            x = x + x.std() * torch.randn_like(x)

        # Normalize
        x = x / x.sum(dim=-1, keepdim=True)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RegressionModelSeparate(nn.Module):
    def __init__(
        self,
        num_bins: int,
        resolution: tuple[int, int],
        output_size: int,
        *,
        min_bin: int | None = None,
        max_bin: int | None = None,
    ):
        super().__init__()
        self.models = []
        for i in range(output_size):
            model = RegressionModel(
                num_bins, resolution, 1, min_bin=min_bin, max_bin=max_bin
            )
            self.models.append(model)
        self.models = nn.ModuleList(self.models)

    def forward(self, x):
        return torch.cat([model(x) for model in self.models], dim=1)
