from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class MaterialConfiguration:
    alias: str
    refractive_index_function: Callable[[float], complex]
    has_absorbing_properties: bool
    thickness_um_lo: float
    thickness_um_hi: float


@dataclass
class DatasetConfiguration:
    refractive_indices_dir: str
    output_path: str
    num_materials_lo: int = 2
    num_materials_hi: int = 5
    num_layers_lo: int = 2
    num_layers_hi: int = 20
    wavelength_um_lo: float = 3
    wavelength_um_hi: float = 14
    wavelength_points: int = 10 * (wavelength_um_hi - wavelength_um_lo) + 1
    num_training_structures: int = 10_000
    num_validation_structures: int = 1_000
    random_seed: Optional[int] = 123
