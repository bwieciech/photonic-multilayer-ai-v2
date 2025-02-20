import glob
import json
import os.path
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Dict, Collection, Tuple, List, Any

import h5py
import numpy as np
import torch.cuda
from numpy._typing import NDArray

from multilayerai.tmm_vectorized.tmm_vectorized import unpolarized_RT_vec
from multilayerai.utils.refractiveindex_info import RefractiveIndexInfoCsvParser


class DatasetType(Enum):
    TRAIN = 0
    VALIDATION = 1


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
    num_materials_lo: int = 2
    num_materials_hi: int = 5
    num_layers_lo: int = 2
    num_layers_hi: int = 20
    wavelength_um_lo: float = 3
    wavelength_um_hi: float = 14
    wavelength_points: int = 10 * (wavelength_um_hi - wavelength_um_lo) + 1
    num_training_structures: int = 100_000
    num_validation_structures: int = 10_000
    random_seed: Optional[int] = 123


def sample_structure(
    available_material_configurations_by_alias: Dict[str, MaterialConfiguration],
    materials_to_indices: Dict[str, int],
    chosen_materials: Collection[str],
    num_layers: int,
) -> Tuple[List[int], List[Callable[[float], complex]], List[float]]:
    output_materials = []
    output_refractive_indices = []
    output_thicknesses = []
    prev_material = None
    while len(output_materials) < num_layers:
        while (material := np.random.choice(chosen_materials)) == prev_material:
            continue
        thickness = np.random.uniform(
            available_material_configurations_by_alias[material].thickness_um_lo,
            available_material_configurations_by_alias[material].thickness_um_hi,
        )
        output_materials.append(materials_to_indices[material])
        output_refractive_indices.append(
            available_material_configurations_by_alias[
                material
            ].refractive_index_function
        )
        output_thicknesses.append(thickness)
        prev_material = material
    return output_materials, output_refractive_indices, output_thicknesses


def pad_with(arr: NDArray[Any], element: Any) -> NDArray[Any]:
    element_column = np.full((arr.shape[0], 1), element)
    return np.hstack((element_column, arr, element_column))


def create_datasets_if_missing(
    file: h5py.File,
    wavelengths_um: List[float],
    config: DatasetConfiguration,
    materials_to_indices: Dict[str, int],
) -> None:
    if file.attrs.get("initialized", False):
        return
    file.attrs["wavelengths_um"] = wavelengths_um
    file.attrs["materials_to_indices"] = json.dumps(materials_to_indices)
    file.attrs["indices_to_materials"] = json.dumps(
        {v: k for k, v in materials_to_indices.items()}
    )
    file.attrs["dataset_type_to_index"] = json.dumps(
        {it.name: it.value for it in DatasetType}
    )

    num_rows = config.num_training_structures + config.num_validation_structures
    print(f"Rows to generate: {num_rows}")

    file.create_dataset(
        "structures_materials",
        (num_rows, 2 + config.num_layers_hi),
        dtype="int8",
        maxshape=(num_rows, 2 + config.num_layers_hi)
    )
    file.create_dataset(
        "structures_thicknesses",
        (num_rows, 2 + config.num_layers_hi),
        dtype="float32",
        maxshape=(num_rows, 2 + config.num_layers_hi)
    )
    file.create_dataset(
        "num_layers",
        (num_rows,),
        dtype="int8",
        maxshape=(num_rows,),
    )
    file.create_dataset(
        "RTA",
        (num_rows, len(wavelengths_um), 3),
        dtype="float32",
        maxshape=(num_rows, len(wavelengths_um), 3),
    )
    file.create_dataset(
        "dataset_type",
        (num_rows,),
        dtype="int8",
        maxshape=(num_rows,),
    )

    file.attrs["offset"] = 0
    file.attrs["initialized"] = True


def generate_dataset(
    dataset_type: DatasetType,
    dataset_config: DatasetConfiguration,
    available_materials_configurations_by_alias: Dict[str, MaterialConfiguration],
    materials_to_indices: Dict[str, int],
    available_materials_aliases: List[str],
    wavelengths_um: Collection[float]
) -> None:
    assert (
        "Air" not in available_materials_aliases
    ), "Air should not be sampled from materials as a non-inf layer"

    with h5py.File("dataset.hdf5", "a") as f:
        create_datasets_if_missing(
            f, wavelengths_um, dataset_config, materials_to_indices
        )
    if dataset_type == DatasetType.TRAIN:
        num_structures = dataset_config.num_training_structures
    elif dataset_type == DatasetType.VALIDATION:
        num_structures = dataset_config.num_validation_structures
    else:
        raise ValueError(f"Dataset type: {dataset_type} not supported")

    print(f"Generating {num_structures} rows for the {dataset_type.name} dataset")
    structure_counts_by_num_layers = Counter(
        np.random.randint(
            dataset_config.num_layers_lo, dataset_config.num_layers_hi, num_structures
        )
    )
    for num_layers, num_structures in structure_counts_by_num_layers.items():
        num_materials = np.random.randint(
            dataset_config.num_materials_lo,
            dataset_config.num_materials_hi,
            num_structures,
        )
        structures_materials, structures_refractive_indices, structures_thicknesses = list(
            zip(
                *[
                    sample_structure(
                        available_materials_configurations_by_alias,
                        materials_to_indices,
                        np.random.choice(
                            available_materials_aliases,
                            num_materials_example,
                            replace=False,
                        ),
                        num_layers,
                    )
                    for num_materials_example in num_materials
                ]
            )
        )
        structures_materials = pad_with(
            np.array(structures_materials),
            element=materials_to_indices["Air"],
        )
        structures_refractive_indices = pad_with(
            np.array(structures_refractive_indices),
            element=available_materials_configurations_by_alias["Air"].refractive_index_function,
        )
        structures_thicknesses = pad_with(
            np.array(structures_thicknesses),
            element=np.inf,
        )
        start = time.perf_counter()
        results = unpolarized_RT_vec(
            structures_refractive_indices,
            structures_thicknesses,
            0,
            wavelengths_um,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(
            f"---> {num_structures} structures with {num_layers} layers: {int(time.perf_counter() - start)}s"
        )
        RTA = np.stack(
            (
                results["R"].squeeze(-1),
                results["T"].squeeze(-1),
                1 - results["R"].squeeze(-1) - results["T"].squeeze(-1)
            ),
            axis=-1
        )
        assert (
            structures_thicknesses.shape[0]
            == structures_refractive_indices.shape[0]
            == RTA.shape[0]
        ), (
            f"# structures_thicknesses rows: {structures_thicknesses.shape[0]}, "
            f"# structures_refractive_indices rows: {structures_refractive_indices.shape[0]}, "
            f"# RTA rows: {RTA.shape[0]}"
        )

        with h5py.File("dataset.hdf5", "a") as f:
            offset = f.attrs["offset"]
            f["structures_materials"][
                offset : offset + num_structures, : structures_thicknesses.shape[1]
            ] = structures_materials
            f["structures_thicknesses"][
                offset : offset + num_structures, : structures_thicknesses.shape[1]
            ] = structures_thicknesses
            f["num_layers"][offset : offset + num_structures] = num_layers
            f["RTA"][offset : offset + num_structures] = RTA
            f["dataset_type"][offset : offset + num_structures] = dataset_type.value

            f.attrs["offset"] += num_structures


if __name__ == "__main__":
    dataset_config = DatasetConfiguration(
        refractive_indices_dir="/home/cupofcoffee/PycharmProjects/photonic-multilayer-ai-v2/assets/refractive_indices"
    )
    if dataset_config.random_seed is not None:
        print(f"Seeding np.random with {dataset_config.random_seed}")
        np.random.seed(dataset_config.random_seed)

    wavelengths_um = np.linspace(
        dataset_config.wavelength_um_lo,
        dataset_config.wavelength_um_hi,
        dataset_config.wavelength_points
    )
    available_materials_configurations_by_alias = {
        (alias := file_path.rsplit("/", 1)[1].split("_")[0]): MaterialConfiguration(
            alias,
            refractive_index_function=(
                parser := RefractiveIndexInfoCsvParser(file_path)
            ).function,
            has_absorbing_properties=(
                has_absorbing_properties := not np.allclose(
                    parser.function(wavelengths_um).imag, 0
                )
            ),
            thickness_um_lo=0.005 if has_absorbing_properties else 0.05,
            thickness_um_hi=0.05 if has_absorbing_properties else 1.0,
        )
        for file_path in glob.glob(
            os.path.join(dataset_config.refractive_indices_dir, "*.csv")
        )
    }
    available_materials_aliases = list(available_materials_configurations_by_alias.keys())
    available_materials_configurations_by_alias["Air"] = MaterialConfiguration(
        "Air",
        lambda wl: 1.00027,
        has_absorbing_properties=False,
        thickness_um_lo=np.inf,
        thickness_um_hi=np.inf,
    )
    materials_to_indices = {
        material: index
        for index, material in enumerate(
            available_materials_configurations_by_alias.keys()
        )
    }
    generate_dataset(
        DatasetType.TRAIN,
        dataset_config,
        available_materials_configurations_by_alias,
        materials_to_indices,
        available_materials_aliases,
        wavelengths_um,
    )
    generate_dataset(
        DatasetType.VALIDATION,
        dataset_config,
        available_materials_configurations_by_alias,
        materials_to_indices,
        available_materials_aliases,
        wavelengths_um,
    )