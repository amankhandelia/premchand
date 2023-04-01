from dataclasses import dataclass


@dataclass
class ModelArchConfig:
    vocab_size: int
    n_embd: int = 512
    n_head: int = 6
    n_layer: int = 14
    dropout: float = 0.2
    block_size: int = 256


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epoch_count: int = 10
    learning_rate: float = 3e-4
    eval_iters: int = 20
    eval_interval: int = 1000


@dataclass
class ModelConfig:
    arch: ModelArchConfig
    training: TrainingConfig
