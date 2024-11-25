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
