from dataclasses import asdict

import mlflow
import optax

import jax.numpy as jnp

import jax

from flax import jax_utils
from flax.training.train_state import TrainState

from jax_smi import initialise_tracking

from loguru import logger

from mingpt.config import ModelConfig, ModelArchConfig, TrainingConfig
from mingpt.dataset import DesiDataset, collate_fn
from data_utils import get_batch, generate
from train_utils import get_model_n_params, update, estimate_loss, save_trained_params

from datasets import load_dataset
from transformers import GPT2Tokenizer

from torch.utils.data import DataLoader, random_split

EXPERIMENT_LABEL = "samachaar_gpt_1"
mlflow.set_experiment("SamachaarGPT")

logger.add("experiment.log", level="INFO", format="{time} {level} {message}")


# Load the dataset
dataset = load_dataset("desiai/samachaar", split="train")

# Define the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Get model and training config
arch_config = ModelArchConfig(tokenizer.vocab_size)
training_config = TrainingConfig()
config = ModelConfig(arch_config, training_config)

# Create the custom dataset
dataset = DesiDataset(dataset, tokenizer, arch_config.block_size)
# Calculate the sizes of the training and validation sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Use random_split to divide the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, arch_config.block_size, True))
val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, arch_config.block_size))




# instansiate the model and get params
initialise_tracking()
gpt, params, dropout_rng = get_model_n_params(config, (2, arch_config.block_size))
max_new_tokens = 10

parameter_count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6
logger.info(f"Number of parameters (in millions): {parameter_count}")


state = TrainState.create(apply_fn=gpt.apply, params=params, tx=optax.adamw(config.training.learning_rate))
state = jax_utils.replicate(state)

p_update = jax.pmap(update, axis_name="batch")

best_val_loss = float("inf")

with mlflow.start_run():
    mlflow.log_params(asdict(config.arch))
    mlflow.log_params(asdict(config.training))
    # mlflow.log_param("token_count", token_count)
    mlflow.log_param("vocab_size", tokenizer.vocab_size)
    batch_count = 0
    for epoch in range(config.training.epoch_count):
        for i, (x, y) in enumerate(train_dataloader):
            
            # sample a batch of data
            x = jnp.asarray(x)
            y = jnp.asarray(y)

            # evaluate the loss
            loss, state = p_update(state, x, y)
            mlflow.log_metric("training loss", float(jnp.mean(loss)), batch_count)

            # every once in a while evaluate the loss on train and val sets
            if batch_count % config.training.eval_interval == 0 or batch_count == config.training.max_iters - 1:
                losses = estimate_loss(val_dataloader, jax_utils.unreplicate(state).params, dropout_rng, config)
                logger.info(f"step {batch_count}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

                mlflow.log_metric("validation loss", losses["validation"], batch_count)
                params = jax_utils.unreplicate(state).params
                if losses["validation"] < best_val_loss:
                    best_val_loss = losses["validation"]
                    save_trained_params(params, "samachaargpt.msgpack")
                    logger.info(f"Updated model at samachaargpt.msgpack with validation loss: {losses['validation']:.4f}")

                context = jnp.zeros((1, 1), dtype=jnp.int32)
                generated_text = decode(generate(context, gpt, params, config, max_new_tokens=max_new_tokens)[0].tolist())
                mlflow.log_text(generated_text, "samples.txt")
                logger.info(generated_text)
            
            batch_count += 1
