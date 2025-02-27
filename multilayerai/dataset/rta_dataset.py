import bisect
import json
from typing import Optional, Tuple, Dict, Any, List

import h5py
import numpy as np
from numpy._typing import NDArray
from torch.utils.data import Dataset

from multilayerai.dataset import DatasetType
from multilayerai.utils.padding import pad_with


class RTADataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: Optional[DatasetType] = None):
        self._dataset_path = dataset_path
        self._reader = h5py.File(dataset_path, "r")
        self._max_layers = self._reader.attrs["max_layers"]  # +2 for the inf layers
        self._materials_to_indices = {
            k: int(v)
            for k, v in json.loads(self._reader.attrs["materials_to_indices"]).items()
        }
        self._padding_idx = len(self._materials_to_indices)
        self._start_of_structure_idx = len(self._materials_to_indices) + 1
        self._end_of_structure_idx = len(self._materials_to_indices) + 2
        self._dataset_type = dataset_type
        (
            self._index_mapping_keys,
            self._index_mapping_values,
        ) = self._create_index_mapping_sorted()
        if dataset_type is not None:
            self._num_rows = self._reader[dataset_type.name].attrs["num_rows"]
        else:
            self._num_rows = self._reader.attrs["num_rows"]

    def __getitem__(
        self, index: int
    ) -> Tuple[NDArray[int], NDArray[float], int, NDArray[float]]:
        (
            dataset_type_key,
            num_layers_key,
            chunk_idx_key,
        ), relative_index = self._get_key_and_relative_index(index)
        num_layers = int(num_layers_key.removeprefix("num_layers="))
        group_reader = self._reader[dataset_type_key][num_layers_key][chunk_idx_key]
        materials = np.concatenate(
            (
                pad_with(
                    group_reader["structures_materials"][
                        relative_index, : num_layers + 2
                    ],
                    l_element=self._start_of_structure_idx,
                    r_element=self._end_of_structure_idx,
                ),
                np.full(self._max_layers - num_layers, self.padding_idx),
            )
        )
        thicknesses = np.concatenate(
            (
                pad_with(
                    np.nan_to_num(
                        group_reader["structures_thicknesses"][
                            relative_index, : num_layers + 2
                        ],
                        posinf=0,
                    ),
                    l_element=0,
                    r_element=0,
                ),
                np.zeros(self._max_layers - num_layers),
            )
        )
        RTA = group_reader["RTA"][relative_index]
        return materials, thicknesses, num_layers, RTA

    @property
    def wavelengths_um(self):
        return self._reader.attrs["wavelengths_um"]

    @property
    def num_tokens(self):
        return len(self._materials_to_indices) + 3

    @property
    def padding_idx(self):
        return self._padding_idx

    def __len__(self) -> int:
        return self._num_rows

    def __getstate__(self) -> Dict[str, Any]:
        self._reader = None
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__ = state
        self._reader = h5py.File(self._dataset_path, "r")

    def _create_index_mapping_sorted(self) -> Tuple[List[int], List[Tuple[str, str]]]:
        dataset_types = (
            (self._dataset_type,) if self._dataset_type is not None else DatasetType
        )
        start_indices = []
        keys = []
        running_mapping_start = 0

        for dataset_type in dataset_types:
            dataset_key = dataset_type.name
            for num_layers_key in self._reader[dataset_key]:
                for chunk_idx_key in self._reader[dataset_key][num_layers_key]:
                    start_indices.append(running_mapping_start)
                    keys.append((dataset_key, num_layers_key, chunk_idx_key))
                    num_rows = self._reader[dataset_key][num_layers_key][
                        chunk_idx_key
                    ].attrs["num_rows"]
                    running_mapping_start += num_rows

        return start_indices, keys

    def _get_key_and_relative_index(self, index: int) -> Tuple[Tuple[str, str], int]:
        pos = bisect.bisect_right(self._index_mapping_keys, index) - 1
        return self._index_mapping_values[pos], index - self._index_mapping_keys[pos]
