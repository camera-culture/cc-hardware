import pickle
from pathlib import Path

import imageio
import numpy as np


class PklWriter:
    def __init__(self, path: Path | str):
        self._path = Path(path)

        if self._path.exists():
            self._path.unlink()

    def write(self, data: dict):
        with open(self._path, "wb") as file:
            pickle.dump(data, file)

    def append(self, data: dict):
        with open(self._path, "ab") as file:
            pickle.dump(data, file)

    def load(self) -> dict:
        with open(self._path, "rb") as file:
            return pickle.load(file)


class VideoWriter:
    def __init__(
        self,
        path: Path | str,
        fps: float,
        frame_size: tuple[int, int],
        flush_interval: int = 10,
    ):
        self._path = Path(path)
        self._fps = fps
        self._frame_size = frame_size
        self._frames = []
        self._flush_interval = flush_interval
        self._frame_count = 0

        if self._path.exists():
            self._path.unlink()

    def append(self, frame: np.ndarray):
        self._frames.append(frame)
        self._frame_count += 1
        if self._frame_count % self._flush_interval == 0:
            self._flush()

    def _flush(self):
        frames = self._frames
        if self._path.exists():
            frames = self._frames + list(imageio.get_reader(self._path))
        imageio.mimwrite(self._path, frames, fps=self._fps)
        self._frames = []

    def close(self):
        if self._frames:
            self._flush()

    def __del__(self):
        self.close()
