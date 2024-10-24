# CC Hardware Algos

This package contains algorithms for processing data.

## Algorithm API

The algorithm API is defined by the abstract base class 
[`Algorithm`](./cc_hardware/algos/algorithm.py). 
All algorithms should inherit from this class.

```python
class Algorithm(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def is_okay(self) -> bool:
        pass

    def close(self):
        pass
```

## Supported Algorithms

In this section, we define all the algorithms currently supported.

### Aruco

The `ArucoLocalizationAlgorithm` class enables support for Aruco markers. It can be 
used, in conjunction with a camera sensor (i.e. 
[`Camera`](../drivers/README.md#camera)), to localize relative to one or more Aruco 
markers. If multiple poses are detected for a single tag, the algorithm will return the 
median of all the poses.


File: [`aruco.py`](./cc_hardware/algos/aruco.py)