import json
import multiprocessing
import os
import uuid
from collections import defaultdict

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from multilayerai.configuration.transformer_rta_predictor_training_configuration import (
    TransformerRTAPredictorTrainingConfiguration,
)
from multilayerai.dataset import RTADataset, DatasetType
from multilayerai.model.transformer.transformer_rta_predictor import (
    TransformerRTAPredictor,
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model(
    model: TransformerRTAPredictor,
    train_dataset: RTADataset,
    val_dataset: RTADataset,
    output_path: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_epochs_threshold: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()  # Mean squared error loss

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(1, int(1.5 * multiprocessing.cpu_count()) - 1),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, int(1.5 * multiprocessing.cpu_count()) - 1),
    )

    best_val_loss = float("inf")
    training_losses = defaultdict(list)
    validation_losses = defaultdict(list)

    # Training
    no_improvement_epochs = 0
    for epoch in range(num_epochs):
        model.train()
        for materials, thicknesses, num_layers, rta in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training process..."
        ):
            materials, thicknesses, rta = (
                materials.to(device),
                thicknesses.to(device),
                rta.to(device),
            )

            optimizer.zero_grad()
            rta_pred = model(materials, thicknesses)
            loss = criterion(rta_pred, rta)
            training_losses[epoch].append(loss.cpu().detach().item())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for materials, thicknesses, num_layers, rta in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Evaluation: Val"
            ):
                materials, thicknesses, rta = (
                    materials.to(device),
                    thicknesses.to(device),
                    rta.to(device),
                )
                rta_pred = model(materials, thicknesses)
                loss = criterion(rta_pred, rta)
                validation_losses[epoch].append(loss.cpu().detach().item())
                val_loss += loss.item() * materials.size(0)
        val_loss /= len(val_dataset)

        train_loss = 0.0
        with torch.no_grad():
            for materials, thicknesses, num_layers, rta in tqdm.tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Evaluation: Train"
            ):
                materials, thicknesses, rta = (
                    materials.to(device),
                    thicknesses.to(device),
                    rta.to(device),
                )
                rta_pred = model(materials, thicknesses)
                loss = criterion(rta_pred, rta)
                training_losses[epoch].append(loss.cpu().detach().item())
                train_loss += loss.item() * materials.size(0)
        train_loss /= len(train_dataset)

        header = f"========== Epoch {epoch + 1}/{num_epochs} =========="
        print(header)
        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")
        print("=" * len(header))

        if val_loss < best_val_loss:
            no_improvement_epochs = 0
            best_val_loss = val_loss
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
            print(f"Saved best model with val Loss: {best_val_loss:.6f}")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs > early_stopping_epochs_threshold:
                print(
                    f"No validation loss improvement for {no_improvement_epochs} epochs, finishing training job."
                )

        # Save metrics
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

    train_dataset = RTADataset(model_training_config.dataset_path, DatasetType.TRAIN)
    val_dataset = RTADataset(model_training_config.dataset_path, DatasetType.VALIDATION)

    model = TransformerRTAPredictor(
        num_tokens=train_dataset.num_tokens,
        embedding_size=model_training_config.embedding_size,
        padding_token_idx=train_dataset.padding_idx,
        num_heads=model_training_config.num_heads,
        ff_hidden_dim=model_training_config.ff_hidden_dim,
        num_encoder_blocks=model_training_config.num_encoder_blocks,
        num_wavelengths=len(train_dataset.wavelengths_um),
        dropout_rate=model_training_config.dropout_rate,
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
        learning_rate=model_training_config.learning_rate,
        weight_decay=model_training_config.weight_decay,
        early_stopping_epochs_threshold=model_training_config.early_stopping_epochs_threshold,
    )
