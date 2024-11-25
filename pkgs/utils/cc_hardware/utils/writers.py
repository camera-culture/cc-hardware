"""
This module provides utility classes for writing data to files.
"""

import pickle
from pathlib import Path

import imageio
import numpy as np


class PklWriter:
    """
    A utility class for writing and reading data to/from pickle files.

    Provides methods to write data, append data, and load single or multiple
    records from a pickle file.

    Example:
        .. code-block:: python

            writer = PklWriter("data.pkl")
            writer.write({"key": "value"})
            writer.append({"another_key": "another_value"})

            data = PklWriter.load("data.pkl")
            all_data = PklWriter.load_all("data.pkl")
    """

    def __init__(self, path: Path | str):
        """
        Initialize the PklWriter.

        Args:
            path (Path | str): The path to the pickle file.
        """
        self._path = Path(path)

        if self._path.exists():
            self._path.unlink()

    def write(self, data: dict):
        """
        Write data to the pickle file, overwriting any existing content.

        Args:
            data (dict): The data to write.
        """
        with open(self._path, "wb") as file:
            pickle.dump(data, file)

    def append(self, data: dict):
        """
        Append data to the pickle file without overwriting.

        Args:
            data (dict): The data to append.
        """
        with open(self._path, "ab") as file:
            pickle.dump(data, file)

    @staticmethod
    def load(path: Path | str) -> dict:
        """
        Load a single record from the pickle file.

        Args:
            path (Path | str): The path to the pickle file.

        Returns:
            dict: The loaded data.
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def load_all(path: Path | str, *, key: str | None = None) -> list[dict]:
        """
        Load all records from the pickle file.

        Args:
            path (Path | str): The path to the pickle file.
            key (str | None): Optional key to extract specific values from each record.

        Returns:
            list[dict]: A list of all records, or specific values if a key is provided.
        """
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
    """
    A utility class for writing video frames to a file.

    Frames are buffered and written to the video file periodically, based on
    the specified flush interval.

    Example:
        .. code-block:: python

            writer = VideoWriter("output.mp4", fps=30)
            for frame in frames:
                writer.append(frame)
            writer.close()
    """

    def __init__(
        self,
        path: Path | str,
        fps: float,
        flush_interval: int = 10,
    ):
        """
        Initialize the VideoWriter.

        Args:
            path (Path | str): The path to the output video file.
            fps (float): The frames per second for the video.
            flush_interval (int): The number of frames to buffer before writing to the
                file.
        """
        self._path = Path(path)
        self._fps = fps
        self._frames = []
        self._flush_interval = flush_interval
        self._frame_count = 0
        self._writer = imageio.get_writer(self._path, fps=self._fps)

    def append(self, frame: np.ndarray):
        """
        Append a video frame to the buffer.

        Args:
            frame (np.ndarray): A single video frame to append.
        """
        self._frames.append(frame)
        self._frame_count += 1
        if self._frame_count % self._flush_interval == 0:
            self._flush()

    def _flush(self):
        """
        Write all buffered frames to the video file.
        """
