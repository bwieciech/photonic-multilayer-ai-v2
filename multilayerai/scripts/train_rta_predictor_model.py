import json
import multiprocessing
import os
import uuid
from collections import defaultdict
from typing import Optional, List, Dict, Union

import numpy as np
import torch
import tqdm
from numpy._typing import NDArray
from torch import nn, Tensor
from torch.utils.data import DataLoader

from multilayerai.configuration.transformer_rta_predictor_training_configuration import (
    TransformerRTAPredictorTrainingConfiguration,
)
from multilayerai.dataset import RTADataset, DatasetType
from multilayerai.model.transformer.transformer_rta_predictor import (
    TransformerRTAPredictor,
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def mse_loss(y_pred: Tensor, y_true: Tensor, n_instances: Optional[int]) -> Tensor:
    if n_instances is None:
        return torch.mean((y_true - y_pred) ** 2)
    y_pred = y_pred.permute(1, 0, 2, 3)
    return torch.mean((y_true - y_pred) ** 2, dim=(1, 2, 3))


def train_model(
    model: TransformerRTAPredictor,
    train_dataset: RTADataset,
    val_dataset: RTADataset,
    output_path: str,
    num_epochs: int,
    batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_epochs_threshold: int,
    cache_data: bool,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if cache_data else max(1, int(1.5 * multiprocessing.cpu_count()) - 1),
    )
    training_validation_loader = DataLoader(
        train_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=0 if cache_data else max(1, int(1.5 * multiprocessing.cpu_count()) - 1),
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=0 if cache_data else max(1, int(1.5 * multiprocessing.cpu_count()) - 1),
    )

    best_val_loss = (
        float("inf")
        if model.n_instances is None
        else np.inf * np.ones(model.n_instances)
    )
    training_losses = defaultdict(list)
    validation_losses = defaultdict(list)

    # Training
    no_improvement_epochs = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for materials, thicknesses, num_layers, rta in tqdm.tqdm(
            training_loader, desc=f"Epoch {epoch}/{num_epochs} - Training..."
        ):
            materials, thicknesses, rta = (
                materials.to(device),
                thicknesses.to(device),
                rta.to(device),
            )

            optimizer.zero_grad()
            rta_pred = model(materials, thicknesses)
            loss = mse_loss(rta_pred, rta, model.n_instances)
            loss.sum().backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = validate(
            validation_loader,
            model,
            validation_losses,
            epoch,
            num_epochs,
            "Validation",
            device,
        )

        train_loss = validate(
            training_validation_loader,
            model,
            training_losses,
            epoch,
            num_epochs,
            "Training",
            device,
        )

        header = f"========== Epoch {epoch}/{num_epochs} =========="
        print(header)
        print(f"Training loss(es): {train_loss}")
        print(f"Validation loss(es): {val_loss}")
        print("=" * len(header))

        payload = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(
            payload,
            os.path.join(output_path, "checkpoint.pth"),
        )

        val_loss_improved = val_loss < best_val_loss
        if model.n_instances is None and val_loss_improved:
            no_improvement_epochs = 0
            best_val_loss = val_loss
        elif model.n_instances is not None and val_loss_improved.any():
            no_improvement_epochs = 0
            best_val_loss[val_loss_improved] = val_loss[val_loss_improved]
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs > early_stopping_epochs_threshold:
                print(
                    f"No validation loss improvement for {no_improvement_epochs} epochs, finishing training job."
                )
                save_metrics(output_path, training_losses, validation_losses)
                break

        save_metrics(output_path, training_losses, validation_losses)


def validate(
    loader: DataLoader,
    model: TransformerRTAPredictor,
    losses_history: Dict[int, Union[List[float], List[NDArray[float]]]],
    epoch: int,
    num_epochs: int,
    dataset_tag: str,
    device: str,
) -> Union[float, NDArray[float]]:
    total_loss = 0.0 if model.n_instances is None else np.zeros(model.n_instances)
    with torch.no_grad():
        for materials, thicknesses, num_layers, rta in tqdm.tqdm(
            loader,
            desc=f"Epoch {epoch}/{num_epochs} - Evaluation - {dataset_tag} Dataset",
        ):
            materials, thicknesses, rta = (
                materials.to(device),
                thicknesses.to(device),
                rta.to(device),
            )
            rta_pred = model(materials, thicknesses)
            loss = mse_loss(rta_pred, rta, model.n_instances)
            if model.n_instances is None:
                losses_history[epoch].append(loss.cpu().item())
                total_loss += loss.item() * materials.size(0)
            else:
                losses_history[epoch].append(loss.cpu().tolist())
                total_loss += loss.cpu().numpy() * materials.size(0)
    return total_loss / len(train_dataset)


def save_metrics(
    output_path: str,
    training_losses: Dict[int, Union[List[float], List[NDArray[float]]]],
    validation_losses: Dict[int, Union[List[float], List[NDArray[float]]]],
):
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(
            {
                "training_losses": training_losses,
                "validation_losses": validation_losses,
            },
            f,
        )


if __name__ == "__main__":
    with open(
        os.path.join(
            CURR_DIR,
            "..",
            "..",
            "configuration",
            "transformer_rta_predictor_training_configuration.json",
        ),
        "r",
    ) as f:
        model_training_config = TransformerRTAPredictorTrainingConfiguration(
            **json.load(f)
        )

    train_dataset = RTADataset(
        model_training_config.dataset_path,
        model_training_config.cache_data,
        DatasetType.TRAIN,
    )
    val_dataset = RTADataset(
        model_training_config.dataset_path,
        model_training_config.cache_data,
        DatasetType.VALIDATION,
    )

    model = TransformerRTAPredictor(
        num_tokens=train_dataset.num_tokens,
        embedding_size=model_training_config.embedding_size,
        padding_token_idx=train_dataset.padding_idx,
        num_heads=model_training_config.num_heads,
        ff_hidden_dim=model_training_config.ff_hidden_dim,
        num_encoder_blocks=model_training_config.num_encoder_blocks,
        num_wavelengths=len(train_dataset.wavelengths_um),
        activation=model_training_config.activation,
        use_layer_norm=model_training_config.use_layer_norm,
    )

    output_path = os.path.join(model_training_config.output_path, uuid.uuid4().hex)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "configuration.json"), "w") as f:
        json.dump(model_training_config.__dict__, f)

    train_model(
        model,
        train_dataset,
        val_dataset,
        output_path=output_path,
        num_epochs=model_training_config.num_epochs,
        batch_size=model_training_config.batch_size,
        validation_batch_size=model_training_config.validation_batch_size,
        learning_rate=model_training_config.learning_rate,
        weight_decay=model_training_config.weight_decay,
        cache_data=model_training_config.cache_data,
        early_stopping_epochs_threshold=model_training_config.early_stopping_epochs_threshold,
    )
