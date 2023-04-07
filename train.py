import os

import mlflow
import optax

import jax.numpy as jnp

import jax

from flax import jax_utils
from flax.training.train_state import TrainState

from jax_smi import initialise_tracking

from loguru import logger

from mingpt.config import load_config
from mingpt.dataset import DesiDataset, collate_fn
from train_utils import get_model_n_params, update, estimate_loss, save_trained_params

from datasets import load_dataset
from tokenizers import Tokenizer

from torch.utils.data import DataLoader, random_split

mlflow.set_experiment("SamachaarGPT")
mlflow.set_tracking_uri("mlruns")
model_directory = "path/to/model_directory"

logger.add("experiment.log", level="INFO", format="{time} {level} {message}")


# Load the dataset
dataset = load_dataset("desiai/samachaar", split="train")

# Define the tokenizer
tokenizer = Tokenizer.from_file("samachaar_tokenizer")
tokenizer.pad_token_id = tokenizer.token_to_id("<PAD>")

# Get model and training config
config = load_config()
config.arch.vocab_size = tokenizer.get_vocab_size()

# Create the custom dataset
dataset = DesiDataset(dataset, tokenizer, config.arch.block_size)
# Calculate the sizes of the training and validation sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Use random_split to divide the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer, config.arch.block_size, True),
    num_workers=16,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer, config.arch.block_size),
)


# instansiate the model and get params
initialise_tracking()
gpt, params, dropout_rng = get_model_n_params(config, (2, config.arch.block_size))
max_new_tokens = 10

parameter_count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6
logger.info(f"Number of parameters (in millions): {parameter_count}")


state = TrainState.create(apply_fn=gpt.apply, params=params, tx=optax.adamw(config.training.learning_rate))
state = jax_utils.replicate(state)

p_update = jax.pmap(update, axis_name="batch", static_broadcasted_argnums=(3,))

best_val_loss = float("inf")

with mlflow.start_run() as run:
    run_name = run.info.run_name
    mlflow.log_params(config.arch.dict())
    mlflow.log_params(config.training.dict())
    mlflow.log_param("vocab_size", config.arch.vocab_size)
    batch_count = 0
    for epoch in range(config.training.epoch_count):
        for i, (x, y) in enumerate(train_dataloader):
            logger.info(f"On Epoch: {epoch} Batch: {batch_count}, Batch Size: {x.shape}")

            # sample a batch of data
            x, y = jnp.asarray(x), jnp.asarray(y)

            # evaluate the loss
            loss, state = p_update(state, x, y, tokenizer.pad_token_id)
            loss = float(jnp.mean(loss))
            logger.info(f"training loss at step {batch_count}: {loss}")
            mlflow.log_metric("training loss", loss, batch_count)

            # every once in a while evaluate the loss on train and val sets
            if batch_count % config.training.eval_interval == 0:
                losses = estimate_loss(
                    val_dataloader,
                    gpt,
                    jax_utils.unreplicate(state).params,
                    tokenizer.pad_token_id,
                    dropout_rng,
                    config,
                )
                logger.info(f"step {batch_count}: val loss {losses:.4f}")

                mlflow.log_metric("validation loss", losses, batch_count)
                params = jax_utils.unreplicate(state).params
                if losses < best_val_loss:
                    best_val_loss = losses
                    model_path = f"{model_directory}/{run_name}/samachaargpt.msgpack"
                    os.makedirs(f"{model_directory}/{run_name}/")
                    save_trained_params(params, model_path)
                    logger.info(f"Updated model at {model_path} with validation loss: {losses:.4f}")

                # context = jnp.zeros((1, 1), dtype=jnp.int32)
                # generated_text = decode(generate(context, gpt, params, config, max_new_tokens=max_new_tokens)[0].tolist())
                # mlflow.log_text(generated_text, "samples.txt")
                # logger.info(generated_text)

            batch_count += 1
