from typing import Any, Optional

import numpy as np
from numpy._typing import NDArray


def pad_with(
    arr: NDArray[Any], l_element: Optional[Any] = None, r_element: Optional[Any] = None
) -> NDArray[Any]:
    if arr.ndim == 1:
        return np.concatenate(([l_element], arr, [r_element]))
    if l_element is not None:
        if isinstance(l_element, np.ndarray):
            assert l_element.ndim == 2 and l_element.shape[0] == arr.shape[0]
            l_element_column = l_element
        else:
            l_element_column = np.full((arr.shape[0], 1), l_element)
        arr = np.hstack((l_element_column, arr))
    if r_element is not None:
        if isinstance(r_element, np.ndarray):
            assert r_element.ndim == 2 and r_element.shape[0] == arr.shape[0]
            r_element_column = r_element
        else:
            r_element_column = np.full((arr.shape[0], 1), r_element)
        arr = np.hstack((arr, r_element_column))
    return arr
