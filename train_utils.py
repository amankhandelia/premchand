from typing import Dict, Tuple, Union, Any

import optax
import torch

import jax.numpy as jnp
import numpy as np

import jax
from jax import lax


from flax.training.train_state import TrainState
from flax.serialization import msgpack_serialize
from flax.core.scope import FrozenVariableDict

from torch.utils.data import DataLoader

from mingpt.gpt import GPTLanguageModel
from mingpt.config import ModelConfig
from data_utils import get_batch


def estimate_loss(dataloader: DataLoader, model: GPTLanguageModel, params, dropout_rng, config: ModelConfig):
    def loss_fn(model: GPTLanguageModel, params, inputs, labels):
        logits = model.apply(params, inputs, rngs={"dropout": dropout_rng})
        B, T, C = logits.shape
        labels = labels.reshape(B * T)
        logits = logits.reshape(B * T, C)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    model.deterministic = True
    losses = np.zeros(config.training.eval_iters)
    for k, (X, Y) in enumerate(dataloader):
        loss = loss_fn(model, params, X, Y)
        losses[k] = loss
        if not (k < config.training.eval_iters):
            break

    model.deterministic = False
    return losses.mean()


def update(state: TrainState, inputs, labels, seed: int = 100):
    dropout_rng = jax.random.PRNGKey(seed)

    def loss_fn(params, labels):
        logits = state.apply_fn(params, inputs, rngs={"dropout": dropout_rng})
        B, T, C = logits.shape
        labels = labels.reshape(B * T)
        logits = logits.reshape(B * T, C)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    val_n_grad = jax.value_and_grad(loss_fn)
    loss, grads = val_n_grad(state.params, labels)
    grads = lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)

    return loss, state


def get_model_n_params(
    config: ModelConfig,
    input_shape: Tuple,
    deterministic: bool = False,
) -> Tuple[GPTLanguageModel, FrozenVariableDict, Any]:
    # prepare rngs
    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}

    # disable dropout when for validation
    deterministic = deterministic

    # create model
    gpt = GPTLanguageModel(
        config.arch.vocab_size,
        config.arch.n_embd,
        config.arch.block_size,
        config.arch.n_layer,
        config.arch.n_head,
        config.arch.dropout,
        deterministic,
    )

    x = jnp.zeros(input_shape)
    params = gpt.init(rngs, x)

    return gpt, params, dropout_rng


def save_trained_params(params, file):
    with open(file, "wb+") as f:
        serialized = msgpack_serialize(params.unfreeze())
        f.write(serialized)
    print(f"Saved successfully to {file}")