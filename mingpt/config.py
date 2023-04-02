from pydantic import BaseSettings
from pathlib import Path
import os


DEFAULT_CONFIG_PATH = "."
ENV_CONFIG_PATH = "CONFIG_PATH"


class ModelArchConfig(BaseSettings):
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    block_size: int


class TrainingConfig(BaseSettings):
    batch_size: int
    epoch_count: int
    learning_rate: float
    eval_iters: int
    eval_interval: int


class ModelConfig(BaseSettings):
    arch: ModelArchConfig
    training: TrainingConfig


def load_config() -> ModelConfig:
    config_path = Path(os.getenv(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH), "config.env")

    if config_path.exists():
        arch_config = ModelArchConfig(_env_file=config_path)
        training_config = TrainingConfig(_env_file=config_path)
        config = ModelConfig(arch=arch_config, training=training_config)
    else:
        raise SystemExit(f"Configuration file '{config_path}' does not exist.")

    return config
