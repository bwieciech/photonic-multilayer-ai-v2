from dataclasses import dataclass
from typing import Union, Collection


@dataclass
class TransformerRTAPredictorTrainingConfiguration:
    dataset_path: str
    output_path: str
    embedding_size: int = 128
    ff_hidden_dim: int = 4 * embedding_size
    num_heads: int = 4
    num_encoder_blocks: Union[int, Collection[int]] = (0, 1, 2, 3, 4)
    num_epochs: int = 64
    batch_size: int = 128
    validation_batch_size: int = 1024
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    early_stopping_epochs_threshold: int = 3
    activation: str = "relu"
    use_layer_norm: bool = True
    cache_data: bool = False
