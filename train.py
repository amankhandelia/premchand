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
from data_utils import generate

from datasets import load_dataset
from tokenizers import Tokenizer

from torch.utils.data import DataLoader, random_split

mlflow.set_experiment("SamachaarGPT")
mlflow.set_tracking_uri("/home/khandelia1000/mlruns")
model_directory = "/home/khandelia1000"

logger.add("experiment.log", level="INFO", format="{time} {level} {message}")


# Load the dataset
dataset = load_dataset("desiai/samachaar", split="train")

# Define the tokenizer
tokenizer = Tokenizer.from_file("devnagari_tokenizer.json")
tokenizer.pad_token_id = tokenizer.token_to_id("<PAD>")

# Get model and training config
config = load_config()
config.arch.vocab_size = tokenizer.get_vocab_size()

# Create the custom dataset
dataset = DesiDataset(dataset, tokenizer, config.arch.block_size)
# Calculate the sizes of the training and validation sets
dataset_size = len(dataset)
train_size = int(0.98 * dataset_size)
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
    collate_fn=lambda b: collate_fn(b, tokenizer, config.arch.block_size, True),
    num_workers=4,
)


# instansiate the model and get params
initialise_tracking()
gpt, params, dropout_rng = get_model_n_params(config, (2, config.arch.block_size))
max_new_tokens = 100

parameter_count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6
logger.info(f"Number of parameters (in millions): {parameter_count}")


state = TrainState.create(apply_fn=gpt.apply, params=params, tx=optax.adamw(config.training.learning_rate))
state = jax_utils.replicate(state)

p_update = jax.pmap(update, axis_name="batch", static_broadcasted_argnums=(3,))
p_estimate_loss = jax.pmap(
    estimate_loss,
    axis_name="batch",
    static_broadcasted_argnums=(3,),
)

best_val_loss = float("inf")

prompt = "सारे जहाँ से अच्छा"
context = jnp.asarray([tokenizer.encode(prompt).ids], dtype=jnp.int32)

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

            if batch_count % 100 == 0:
                loss = float(jnp.mean(loss))
                logger.info(f"training loss at step {batch_count}: {loss}")
                mlflow.log_metric("training loss", loss, batch_count)

            # every once in a while evaluate the loss on train and val sets
            if batch_count % config.training.eval_interval == 0:
                gpt.deterministic = True
                params = jax_utils.unreplicate(state).params
                losses = []
                for k, (X, Y) in enumerate(val_dataloader):
                    if k >= config.training.eval_iters:
                        break
                    loss = p_estimate_loss(state, X, Y, tokenizer.pad_token_id)
                    losses.append(float(jnp.mean(loss)))

                gpt.deterministic = False
                val_loss = float(jnp.mean(jnp.asarray(losses)))

                logger.info(f"step {batch_count}: val loss {val_loss:.4f}")

                mlflow.log_metric("validation loss", val_loss, batch_count)

                if val_loss < best_val_loss:
                    best_val_loss = losses
                    model_path = f"{model_directory}/{run_name}/samachaargpt.msgpack"
                    os.makedirs(f"{model_directory}/{run_name}/", exist_ok=True)
                    save_trained_params(params, model_path)
                    logger.info(f"Updated model at {model_path} with validation loss: {val_loss:.4f}")

                generated_tokens = generate(
                    context, gpt, params, config, max_new_tokens, dropout_rng, tokenizer.pad_token_id
                )
                first_zero_index = int(jnp.where(generated_tokens[0] == 0)[0][0])
                generated_text = tokenizer.decode(generated_tokens[0, :first_zero_index].tolist())
                logger.info(f"Sampled text at step: {batch_count} is {generated_text}")

            batch_count += 1
