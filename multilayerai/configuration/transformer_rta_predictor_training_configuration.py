from dataclasses import dataclass


@dataclass
class TransformerRTAPredictorTrainingConfiguration:
    dataset_path: str
    output_path: str
    embedding_size: int = 128
    ff_hidden_dim: int = 4 * embedding_size
    num_heads: int = 4
    num_encoder_blocks: int = 1
    dropout_rate: float = 0.0
    num_epochs: int = 64
    batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
