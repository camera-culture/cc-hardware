from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import RegressionModelSeparate
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from cc_hardware.tools.cli import register_cli, run_cli
from cc_hardware.utils import get_logger
from cc_hardware.utils.file_handlers import PklReader


class HistogramDataset(Dataset):
    def __init__(self, pkl: Path, predict_magnitude: bool = False, merge: bool = False):
        self.data = PklReader.load_all(pkl)
        inputs = dict(
            histogram=[],
            position=[],
        )
        for d in self.data:
            if "has_masks" in d and not d["has_masks"]:
                d["position"] = [0, 0, 0]
                continue
            if "histogram" not in d or "position" not in d:
                continue
            inputs["histogram"].append(torch.tensor(d["histogram"]))
            inputs["position"].append(torch.tensor(d["position"]))

        self.original_targets = torch.stack(inputs["position"]).float()
        self.inputs = torch.stack(inputs["histogram"]).float()
        self.targets = torch.stack(inputs["position"]).float()

        if predict_magnitude:
            self.targets = torch.linalg.norm(self.targets, dim=1, keepdim=True)

        if merge:
            self.inputs = self.inputs.sum(dim=(1), keepdim=True)

    def __len__(self):
        return len(self.original_targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class HistogramPolarDataset(HistogramDataset):
    """Same as HistogramDataset, but converts position to polar coordinates."""

    def __init__(self, pkl: Path, predict_magnitude: bool = False, merge: bool = False):
        super().__init__(pkl, predict_magnitude=predict_magnitude, merge=merge)
        self.targets = torch.stack(
            [
                torch.linalg.norm(self.original_targets, dim=1),
                torch.atan2(self.original_targets[:, 1], self.original_targets[:, 0]),
            ],
            dim=1,
        )

        if predict_magnitude:
            self.targets = self.targets[:, 0].unsqueeze(1)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    predict_magnitude: bool = False,
    is_polar: bool = False,
):
    model.eval()
    actual = []
    predictions = []
    with torch.no_grad():
        for histograms, targets in loader:
            outputs = model(histograms)
            actual.extend(targets.numpy())
            predictions.extend(outputs.numpy())

    actual = torch.tensor(actual)
    predictions = torch.tensor(predictions)
    if is_polar and not predict_magnitude:
        actual = torch.stack(
            [
                actual[:, 0] * torch.cos(actual[:, 1]),
                actual[:, 0] * torch.sin(actual[:, 1]),
            ],
            dim=1,
        )
        predictions = torch.stack(
            [
                predictions[:, 0] * torch.cos(predictions[:, 1]),
                predictions[:, 0] * torch.sin(predictions[:, 1]),
            ],
            dim=1,
        )

    plt.figure(figsize=(10, 6))
    if predict_magnitude:
        plt.scatter(actual, predictions, label="Magnitude")
        plt.xlabel("Actual Magnitude")
        plt.ylabel("Predicted Magnitude")
        plt.title("Actual vs Predicted Magnitudes")
    else:
        plt.scatter(actual[:, 0], actual[:, 1], label="Actual")
        plt.scatter(predictions[:, 0], predictions[:, 1], label="Predictions")
        # Connect a line between actual and predicted positions
        for a, p in zip(actual, predictions):
            plt.plot([a[0], p[0]], [a[1], p[1]], "y--", alpha=0.5)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Actual vs Predicted Positions")
    plt.legend()
    plt.grid()
    plt.show()


@register_cli(simple=True)
def train_model(
    pkl: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    predict_magnitude: bool = False,
    merge: bool = False,
    min_bin: int = 30,
    max_bin: int = 70,
    save_model: bool = True,
    output_stem: str | None = None,
    load_model: Path | None = None,
):
    torch.manual_seed(0)
    dataset = HistogramPolarDataset(
        pkl, predict_magnitude=predict_magnitude, merge=merge
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(dataset, batch_size=batch_size)

    output_size = 1 if predict_magnitude else len(dataset.targets[0])
    resolution = (3, 3) if not merge else (1, 1)
    model = RegressionModelSeparate(
        128, resolution, output_size, min_bin=min_bin, max_bin=max_bin
    )
    if load_model:
        get_logger().info(f"Loading model from {load_model}")
        model.load_state_dict(torch.load(load_model))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    model.train()
    pbar = tqdm(range(epochs), desc="Training Progress", leave=False)
    for epoch in pbar:
        train_loss = 0.0
        for histograms, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(histograms)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for histograms, targets in val_loader:
                outputs = model(histograms)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        model.train()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        pbar.set_postfix(
            {
                "Train Loss": f"{train_losses[-1]:.4f}",
                "Validation Loss": f"{val_losses[-1]:.4f}",
            }
        )
        # if epoch % 10 == 0 or epoch == epochs - 1:
        #     get_logger().info(
        #         f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}"
        #     )

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()

    # Evaluation plot
    evaluate_model(
        model,
        val_loader,
        predict_magnitude=predict_magnitude,
        is_polar=isinstance(dataset, HistogramPolarDataset),
    )

    if save_model:
        output_path = pkl.with_name((output_stem or pkl.stem) + "_model.pth")
        torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    run_cli(train_model)
