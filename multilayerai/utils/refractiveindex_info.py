import functools
import os
from typing import Iterable, TextIO, Callable, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@functools.cache
class RefractiveIndexInfoCsvParser:
    def __init__(self, path: str):
        with open(path, "r") as file:
            self._function = self._as_function(file)

    @property
    def function(self) -> Callable[[float], complex]:
        """
        :return: function that takes wavelength in um and returns complex refractive index
        """
        return self._function

    @staticmethod
    def _as_function(file: TextIO) -> Callable[[float], complex]:
        df_n = RefractiveIndexInfoCsvParser._process_lines(file)
        df_k = RefractiveIndexInfoCsvParser._process_lines(file)

        fun_n = interp1d(
            df_n["wl"].to_numpy().astype("double"), df_n["n"].to_numpy(), kind="linear"
        )
        if df_k is not None:
            fun_k = interp1d(
                df_k["wl"].to_numpy().astype("double"),
                df_k["k"].to_numpy(),
                kind="linear",
            )
        else:
            fun_k = lambda x: np.zeros_like(x) if isinstance(x, Iterable) else 0
        return lambda x: fun_n(x) + 1j * fun_k(x)

    @staticmethod
    def _process_lines(file: TextIO) -> Optional[pd.DataFrame]:
        processed_lines = []

        columns_line = next(file, None)
        if columns_line is None:
            return None

        columns = [c.strip() for c in columns_line.split(",")]
        if len(columns) != 2:
            raise ValueError(
                "Expected header to be of form (wavelength, n or k), where n_tilde = n + ik"
            )

        line = next(file, None)
        while line is not None and line.strip() != "":
            line_parts = line.strip().split(",")
            if len(line_parts) != 2:
                raise ValueError(
                    "Expected line to be of form (wavelength, n or k) values, where n_tilde = n + ik"
                )
            processed_lines.append([float(it) for it in line_parts])
            line = next(file, None)
        return pd.DataFrame(processed_lines, columns=columns).drop_duplicates("wl")

    @staticmethod
    def get_alias(path: str) -> str:
        formula, _ = os.path.basename(path).split("_", 1)
        return formula
