import torch
from torch.utils.data import DataLoader, Dataset, random_split

from cc_hardware.drivers.spads import SPADDataType


class CaptureDataset(Dataset):
    def __init__(self, captures, h, w, bins, window: int = 1):
        """
        Args:
            captures (list[dict]): raw frames
            h, w, bins (int): histogram dimensions
            window (int): number of neighbouring frames to average
                          (1 → no averaging)
        """
        self.cap = captures
        self.h, self.w, self.bins = h, w, bins
        self.win = max(window, 1)

    def __len__(self):  # noqa: D401
        return len(self.cap)

    def _avg_hist(self, idx: int) -> torch.Tensor:
        half = self.win // 2
        lo = max(0, idx - half)
        hi = min(len(self.cap), idx + half + 1)
        h_arr = [
            torch.tensor(self.cap[i][SPADDataType.HISTOGRAM], dtype=torch.float32)
            for i in range(lo, hi)
        ]
        return torch.mean(torch.stack(h_arr), dim=0)

    def __getitem__(self, idx):
        f = self.cap[idx]
        pos = torch.tensor([f["pos"]["x"], f["pos"]["y"]], dtype=torch.float32)
        hist = (
            self._avg_hist(idx)
            if self.win > 1
            else torch.tensor(f[SPADDataType.HISTOGRAM], dtype=torch.float32)
        )
        if hist.ndim == 1:
            hist = hist.view(self.h, self.w, self.bins)
        return pos, hist


def collate(batch):
    pos, hist = zip(*batch)
    return torch.stack(pos), torch.stack(hist)


def create_dataloaders(captures, split: float = 0.8, batch_size: int = 32, **kwargs):
    """
    Create DataLoader for training and validation datasets.

    Args:
        captures (list[dict]): raw frames
        split (float): fraction of data for training (0.8 → 80% train, 20% val)
        batch_size (int): size of each batch
        **kwargs: additional arguments for CaptureDataset

    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = CaptureDataset(captures, **kwargs)
    train_size = int(len(dataset) * split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    return train_loader, val_loader
