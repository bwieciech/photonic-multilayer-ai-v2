from typing import Any, Optional

import numpy as np
from numpy._typing import NDArray


def pad_with(
    arr: NDArray[Any], l_element: Optional[Any] = None, r_element: Optional[Any] = None
) -> NDArray[Any]:
    if arr.ndim == 1:
        return np.concatenate(([l_element], arr, [r_element]))
    if l_element is not None:
        l_element_column = np.full((arr.shape[0], 1), l_element)
        arr = np.hstack((l_element_column, arr))
    if r_element is not None:
        r_element_column = np.full((arr.shape[0], 1), r_element)
        arr = np.hstack((arr, r_element_column))
    return arr
