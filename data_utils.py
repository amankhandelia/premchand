from typing import Dict, Tuple, Callable

from flax.core.scope import FrozenVariableDict


import jax
import jax.numpy as jnp

from einops import rearrange

import torch

from histr import Shabdansh

from mingpt.train import GraphemeVocab
from mingpt.gpt import GPTLanguageModel
from mingpt.config import ModelConfig


def generate(
    idx: jnp.ndarray, model: GPTLanguageModel, params: FrozenVariableDict, config: ModelConfig, max_new_tokens: int, dropout_rng
):
    model.deterministic = True
    # idx is (B, T) array of indices in the current context
    key = jax.random.PRNGKey(4223)

    for i in range(max_new_tokens):
        idx_cond = idx[:, -config.arch.block_size :]

        # get the predictions
        logits = model.apply(params, idx_cond, rngs={"dropout": dropout_rng})

        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)

        # apply softmax to get probabilities
        probs = jax.nn.softmax(logits, axis=-1)  # (B, C)
        # probs = np.asarray(probs)

        # sample from the distribution
        sample_key = jax.random.fold_in(key, i)
        idx_next = jax.random.choice(sample_key, probs.shape[-1], shape=(1, 1), p=jnp.reshape(probs, (-1)))

        # append sampled index to the running sequence
        idx = jnp.concatenate([idx, idx_next], axis=1)  # (B, T+1)

    model.deterministic = False
    return idx


def load_data(data_path: str) -> Tuple[str, GraphemeVocab]:
    articles = open(data_path, "r").readlines()
    vocab = GraphemeVocab()
    vocab.build_vocab(articles)
    return " ".join(articles), vocab


def get_batch(split, data: Dict[str, torch.tensor], block_size: int, batch_size: int, for_pmap: bool = False):
    # generate a small batch of data of inputs x and targets y
    data = data[split]
    if for_pmap:
        batch_size = jax.device_count() * batch_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.numpy(), y.numpy()
    if for_pmap:
        x = rearrange(
            x,
            "(device_count batch_size) block_size -> device_count batch_size block_size",
            device_count=jax.device_count(),
        )
        y = rearrange(
            y,
            "(device_count batch_size) block_size -> device_count batch_size block_size",
            device_count=jax.device_count(),
        )
    return x, y


def get_data_dict(text: str, encode: Callable) -> Dict[str, torch.tensor]:
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    data_dict = {"train": train_data, "validation": val_data}

    return data_dict


def get_encoder_decoder(vocab: GraphemeVocab) -> Tuple[Callable, Callable]:
    # create a mapping from characters to integers
    stoi = vocab.stoi
    itos = vocab.itos

    def encode(s):
        return [stoi[c] for c in Shabdansh(s)]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    return encode, decode
