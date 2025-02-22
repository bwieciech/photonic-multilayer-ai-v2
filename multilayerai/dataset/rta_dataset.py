import json
from functools import cache
from typing import Optional, Tuple, Dict, Any

import h5py
import numpy as np
from numpy._typing import NDArray
from torch.utils.data import Dataset

from multilayerai.dataset import DatasetType


class RTADataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: Optional[DatasetType] = None):
        self._dataset_path = dataset_path
        self._reader = h5py.File(dataset_path, "r")
        offset = self._reader.attrs["offset"]
        if dataset_type is not None:
            dataset_type_index = json.loads(
                self._reader.attrs["dataset_type_to_index"]
            )[dataset_type.name]
            (self._index_mapping,) = np.where(
                self._reader["dataset_type"][:offset] == dataset_type_index
            )
            self._num_rows = len(self._index_mapping)
        else:
            self._index_mapping = None
            self._num_rows = offset

    def __getitem__(
        self, i: int
    ) -> Tuple[NDArray[int], NDArray[float], int, NDArray[float]]:
        materials = self._reader["structures_materials"][i]
        thicknesses = self._reader["structures_thicknesses"][i]
        num_layers = self._reader["num_layers"][i] + 2
        RTA = self._reader["RTA"][i]
        return materials, thicknesses, num_layers, RTA

    @property
    @cache
    def wavelengths_um(self):
        return self._reader.attrs["wavelengths_um"]

    def __len__(self) -> int:
        return self._num_rows

    def __getstate__(self) -> Dict[str, Any]:
        self._reader = None
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__ = state
        self._reader = h5py.File(self._dataset_path, "r")
