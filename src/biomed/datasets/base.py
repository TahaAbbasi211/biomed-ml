from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Contract for every dataset in the project.

    load() must return (X, y, groups):
      X      - 2D numpy array, one row per sample
      y      - 1D numpy array of labels
      groups - 1D array saying which recording/patient each row came from
               (or None if unknown), used for leakage-free splitting
    """

    name = "base"

    @abstractmethod
    def load(self):
        raise NotImplementedError
