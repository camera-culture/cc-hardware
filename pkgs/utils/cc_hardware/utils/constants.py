"""Constants."""

import torch

C = 3e8
"""Speed of light in m/s"""

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Default device for PyTorch operations, using CUDA if available, otherwise CPU."""
