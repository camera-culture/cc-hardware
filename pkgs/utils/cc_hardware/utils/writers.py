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

    @staticmethod
    def load(path: Path | str) -> dict:
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def load_all(path: Path | str, *, key: str | None = None) -> list[dict]:
        data = []
        with open(path, "rb") as file:
            try:
                while True:
                    entry = pickle.load(file)
                    data.append(entry if key is None else entry[key])
            except EOFError:
                pass
        return data


class VideoWriter:
    def __init__(
        self,
        path: Path | str,
        fps: float,
        flush_interval: int = 10,
    ):
        self._path = Path(path)
        self._fps = fps
        self._frames = []
        self._flush_interval = flush_interval
        self._frame_count = 0
        self._writer = imageio.get_writer(self._path, fps=self._fps)

    def append(self, frame: np.ndarray):
        self._frames.append(frame)
        self._frame_count += 1
        if self._frame_count % self._flush_interval == 0:
            self._flush()

    def _flush(self):
        for frame in self._frames:
            self._writer.append_data(frame)
        self._frames = []

    def close(self):
        if self._frames:
            self._flush()
        self._writer.close()

    def __del__(self):
        self.close()
