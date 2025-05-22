"""Datasets and data processing utils for modeling"""

from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

from cc_hardware.utils.file_handlers import PklReader


class HistogramDataset(Dataset):
    """
    Histogram dataset for model training and analysis. Contains a corresponding list of inputs (histograms) and targets.
    Can manipulate inputs and targets without affecting raw data.
    """

    def __init__(self, pkl_path: Path = None, rolling_window=1):
        """
        Initializes the HistogramDataset.

        Args:
        pkl_path (Path): Path to the PKL file containing the histogram data.
        rolling_window (int): Number of samples to use for rolling mean smoothing. This is applied to raw inputs. Default is 1.
        predict_magnitude (bool): Whether to predict the magnitude of the position. 
            Computes Euclidean distance from the origin. Default is False.
        merge (bool): Whether to merge the input and target data. Default is False.
        """
        if pkl_path is None:
            self.data = []
            self.inputs = []
            self.targets = []
            return

        self.data = PklReader.load_all(pkl_path)
        inputs = dict(
            histogram=[],
            position=[],
        )
        for d in self.data:
            if "has_masks" in d and not d["has_masks"]:
                d["position"] = [0, 0, 0]
                print("mask continue")
                continue
            if "histogram" in d:
                inputs["histogram"].append(torch.tensor(d["histogram"]))
            if "position" in d:
                pos = d["position"]
                inputs["position"].append(torch.tensor((pos["x"], pos["y"])))
            else:
                print("Warning: no position")
                inputs["position"].append(torch.tensor((0, 0)))
        
        if len(inputs["histogram"][0].shape) == 2:
            # single capture per location
            # reading input as list of location samples: (pixels, bins)
            # reading targets as list of location samples: ((x, y) position)
            self.raw_inputs = torch.stack(inputs["histogram"]).float()
            num_samples = self.raw_inputs.shape[0]
            width = np.sqrt(self.raw_inputs.shape[1]).astype(int)
            height = width
            bins = self.raw_inputs.shape[2]
            self.raw_inputs = torch.reshape(self.raw_inputs, (num_samples, width, height, bins))
            self.raw_targets = torch.stack(inputs["position"]).float()
        elif len(inputs["histogram"][0].shape) == 3:
            # multiple captures per location:
            # reading input as list of location samples: (captures per location, pixels, bins)
            # reading targets as list of location samples: ((x, y) position)
            self.raw_inputs = torch.stack(inputs["histogram"]).float()  # (locations, captures, pixels, bins)
            samples_per_location = inputs["histogram"][0].shape[0]
            if rolling_window is not 1:
                sliding_mean = np.array(
                    [self.raw_inputs[:, j - rolling_window + 1: j + 1, :, :].mean(axis=1)
                        for j in range(rolling_window - 1, samples_per_location)
                    ]).swapaxes(0, 1)
                self.raw_inputs = torch.tensor(sliding_mean)
            samples_per_location = self.raw_inputs.shape[1]

            self.raw_inputs = torch.reshape(self.raw_inputs, 
                (self.raw_inputs.shape[0] * self.raw_inputs.shape[1], self.raw_inputs.shape[2], self.raw_inputs.shape[3]))
            num_samples = self.raw_inputs.shape[0]
            width = np.sqrt(self.raw_inputs.shape[1]).astype(int)
            height = width
            bins = self.raw_inputs.shape[2]
            self.raw_inputs = torch.reshape(self.raw_inputs, (num_samples, width, height, bins))
            self.raw_targets = torch.stack(inputs["position"], dim=0).float()  # (location samples, (x, y) position)
            self.raw_targets = torch.repeat_interleave(self.raw_targets, samples_per_location, dim=0)

        self.START_BIN = 0
        self.END_BIN = bins
        self.inputs = self.raw_inputs
        self.targets = self.raw_targets

    def set_start_bin(self, start_bin: int):
        """
        Sets the start bin for the input data.
        """
        self.START_BIN = start_bin
        self.inputs = self.raw_inputs[:, :, :, self.START_BIN:self.END_BIN]

    def set_end_bin(self, end_bin: int):
        """
        Sets the end bin for the input data.
        """
        self.END_BIN = end_bin
        self.inputs = self.raw_inputs[:, :, :, self.START_BIN:self.END_BIN]

    def get_raw_bin_num(self):
        """
        Returns the number of bins in the raw input data (with no transformations applied).

        Returns:
            int: The number of bins in the raw input data.
        """
        return self.raw_inputs.shape[3]

    def get_bin_num(self):
        """
        Returns the number of bins in the input data.

        Returns:
            int: The number of bins in the input data.
        """
        return self.inputs.shape[3]
    
    def augment(self, factor: int, std_multiplier: float = 1.0, group_by_targets: bool = True):
        """
        Augments the dataset by repeating the inputs and targets a given number of times.

        Args:
            factor (int): The number of times to repeat the inputs and targets.
            std_multiplier (float): Number of standard deviations to generate augmented sample from.
        """
        if group_by_targets:
            augmented_inputs = []
            augmented_targets = []
            groups = self.targets.unique(dim=0)
            for group in groups:
                group_mask = torch.all(self.targets == group, dim=1)
                group_inputs = self.inputs[group_mask]
                group_targets = self.targets[group_mask]
                group_inputs = group_inputs.repeat_interleave(factor, dim=0)
                group_inputs += torch.normal(torch.zeros_like(group_inputs), group_inputs.std(dim=0) * std_multiplier)
                group_targets = group_targets.repeat_interleave(factor, dim=0)
                augmented_inputs.append(group_inputs)
                augmented_targets.append(group_targets)
            self.inputs = torch.cat(augmented_inputs, dim=0)
            self.targets = torch.cat(augmented_targets, dim=0)
        else:
            std = self.inputs.std(dim=0)
            self.inputs = self.inputs.repeat_interleave(factor, dim=0)
            self.inputs += torch.normal(torch.zeros_like(self.inputs), std * std_multiplier)
            self.targets = self.targets.repeat_interleave(factor, dim=0)

    def get_mean_capture(self):
        """
        Gets the mean of all input captures in the dataset.

        Returns:
            torch.Tensor: The mean capture.
        """
        return self.inputs.mean(dim=0)
    
    def set_zero(self, zero: torch.Tensor):
        """
        Sets the empty capture of the dataset, subtracting it from all inputs.
        """
        self.inputs = self.inputs - zero
    
    def clip_negative(self):
        """
        Clips negative input values to zero.
        """
        self.inputs[self.inputs < 0] = 0

    def reset_transformations(self):
        """
        Resets the transformations applied to the dataset.
        """
        self.inputs = self.raw_inputs
        self.targets = self.raw_targets

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns: 
            int: The number of samples in the dataset
        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Returns the item at index position (idx).

        Returns: 
            (torch.Tensor, torch.Tensor): The (input, target) at index position (idx)
        """
        return self.inputs[idx], self.targets[idx]
    
    def __add__(self, other):
        """
        Joins two datasets. Use only on datasets confirmed to contain data of the same dimension.
        """
        combined_dataset = HistogramDataset()
        combined_dataset.raw_inputs = torch.cat((self.raw_inputs, other.raw_inputs), dim=0)
        combined_dataset.raw_targets = torch.cat((self.raw_targets, other.raw_targets), dim=0)
        combined_dataset.START_BIN = self.START_BIN
        combined_dataset.END_BIN = self.END_BIN
        combined_dataset.inputs = combined_dataset.raw_inputs[:, :, :, self.START_BIN:self.END_BIN]
        combined_dataset.targets = combined_dataset.raw_targets
        return combined_dataset


