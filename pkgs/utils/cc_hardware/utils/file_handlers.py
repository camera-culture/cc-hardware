"""
This module provides utility classes for writing data to files.
"""

import pickle
from pathlib import Path
from typing import Any, Callable

import cloudpickle as pickle
import imageio
import numpy as np

# ==================


class PklHandler:
    """
    A utility class for writing and reading data to/from pickle files.

    Provides methods to write data, append data, and load single or multiple
    records from a pickle file. It also supports indexing for random access
    to records without loading the entire file into memory.

    Keyword Args:
        overwrite (bool): If True, overwrite existing file when writing. Defaults to
            True.

    Example:
        .. code-block:: python

            handler = PklHandler("data.pkl")
            handler.write({"key": "value"})
            handler.append({"another_key": "another_value"})

            record = handler.load()
            print(record) # {"key": "value"}
            record = handler.load(1)
            print(record) # {"another_key": "another_value"}
    """

    def __init__(
        self,
        path: Path | str,
        *,
        overwrite: bool = True,
        **update_index_kwargs,
    ):
        """
        Initialize the PklHandler.

        Args:
            path (Path | str): The path to the pickle file.
            overwrite (bool): If True, overwrite existing file when writing. Defaults
                to True.
        """
        self._path = Path(path)
        self._index = []

        if overwrite:
            self._path.unlink(missing_ok=True)
            self._path.touch()

        self.update_index(**update_index_kwargs)

    def update_index(self, *, prune_fn: Callable[[Any], bool] | None = lambda _: False):
        """
        Build an index of record positions in the pickle file for random access.

        This method scans the pickle file and records the file positions
        where each record starts. The positions are stored in self._index.

        Keyword Args:
            prune_fn (Callable[[Any], bool]): Optional function to filter records when
                building the index. The function should take a record as input and
                return True to exclude the record from the index.
                Defaults to lambda _: False.
        """

        self._index = []
        with open(self._path, "rb") as file:
            while True:
                try:
                    pos = file.tell()
                    if prune_fn(pickle.load(file)):
                        continue
                    self._index.append(pos)
                except EOFError:
                    break

    def write(self, data: Any):
        """
        Write data to the pickle file, overwriting any existing content.

        Args:
            data (Any): The data to write.
        """
        with open(self._path, "wb") as file:
            pickle.dump(data, file)

    def append(self, data: Any):
        """
        Append data to the pickle file without overwriting.

        Args:
            data (Any): The data to append.
        """
        with open(self._path, "ab") as file:
            pickle.dump(data, file)

    def load(self, index: int | None = None) -> Any:
        """
        Load the first record from the pickle file.

        Returns:
            Any: The loaded record.
        """
        if index is not None:
            with open(self._path, "rb") as file:
                file.seek(self._index[index])
                return pickle.load(file)
        else:
            with open(self._path, "rb") as file:
                return pickle.load(file)

    @staticmethod
    def load_all(path: Path | str, *, key: str = None) -> list[Any]:
        """
        Load all records from the pickle file.

        Args:
            path (Path | str): The path to the pickle file.
            key (str | None): Optional key to extract specific values from each record.

        Returns:
            list[Any]: A list of all records, or specific values if a key is provided.
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

    def __len__(self) -> int:
        """
        Get the number of records in the pickle file.

        Returns:
            int: The number of records.
        """
        return len(self._index)


class PklReader(PklHandler):
    def __init__(self, path: Path | str):
        super().__init__(path, overwrite=False)


# ==================


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
        for frame in self._frames:
            self._writer.append_data(frame)
        self._frames = []

    def close(self):
        """
        Flush any remaining frames and close the video writer.
        """
        if self._frames:
            self._flush()
        self._writer.close()

    def __del__(self):
        """
        Ensure that the video writer is properly closed when the object is deleted.
        """
        self.close()
