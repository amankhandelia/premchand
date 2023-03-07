import csv
import logging
from typing import Dict, Optional, Tuple, Callable, Union, Any

import jax
import jax.numpy as jnp
from jax import lax, jit

import numpy as np
import optax
import torch

from einops import rearrange

from flax import jax_utils
from flax import linen as nn
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState

from histr import Shabdansh

from jax_smi import initialise_tracking

from mingpt.train import GraphemeVocab

MAX_FIELD_SIZE = 100000000  # 100 MB
EXPERIMENT_LABEL = "test_run_3"

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define log format
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# Define file handler
file_handler = logging.FileHandler(f"gpt_jax_{EXPERIMENT_LABEL}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Define console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class Head(nn.Module):
    """one head of self-attention"""

    head_size: int
    dropout_rate: float
    block_size: int
    deterministic: Optional[bool] = None

    def setup(self):
        self.key = nn.Dense(self.head_size, use_bias=False)
        self.query = nn.Dense(self.head_size, use_bias=False)
        self.value = nn.Dense(self.head_size, use_bias=False)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.tril = jnp.tril(jnp.ones((self.block_size, self.block_size)))

    def __call__(self, x, deterministic: bool = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        key = self.key(x)  # (B,T,hs)
        query = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        key = rearrange(key, "batch_size seq_len dim -> batch_size dim seq_len")
        wei = jnp.matmul(query, key) * jnp.sqrt(self.head_size)  # (B, T, T)
        wei = jnp.where(self.tril[:T, :T] == 0, jnp.full((T, T), -1e9), wei)  # (B, T, T)
        wei = nn.softmax(wei, axis=-1)  # (B, T, T)
        wei = self.dropout(wei, deterministic)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = jnp.matmul(wei, v)  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    num_heads: int
    head_size: int
    n_embd: int
    dropout_rate: float
    block_size: int
    deterministic: Optional[bool] = None

    def setup(self):
        self.heads = [
            Head(self.head_size, self.dropout_rate, self.block_size, self.deterministic) for _ in range(self.num_heads)
        ]
        self.proj = nn.Dense(self.n_embd, use_bias=True)
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, n_embd)
        heads_out = jnp.concatenate([head(x) for head in self.heads], axis=-1)
        proj_out = self.proj(heads_out)
        out = self.dropout_layer(proj_out, deterministic=deterministic)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    n_embd: int
    dropout: float
    deterministic: Optional[bool] = None

    def setup(self):
        self.fc1 = nn.Dense(4 * self.n_embd)
        self.fc2 = nn.Dense(self.n_embd)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        out = self.fc1(x)
        out = nn.relu(out)
        out = self.fc2(out)
        out = self.dropout_layer(out, deterministic)
        return out


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    n_embd: int
    n_head: int
    block_size: int
    dropout: float
    deterministic: Optional[bool] = None

    def setup(self):
        head_size = self.n_embd // self.n_head
        self.mha = MultiHeadAttention(
            self.n_head, head_size, self.n_embd, self.dropout, self.block_size, self.deterministic
        )
        self.ffwd = FeedForward(self.n_embd, self.dropout, self.deterministic)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    vocab_size: int
    n_embd: int
    block_size: int
    n_layer: int
    n_head: int
    dropout: float
    deterministic: Optional[bool] = None

    def setup(self):
        self.token_embedding_table = nn.Embed(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embed(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(
            [
                Block(self.n_embd, self.n_head, self.block_size, self.dropout, self.deterministic)
                for _ in range(self.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Dense(self.vocab_size)

    def __call__(self, idx, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(jnp.arange(T, dtype=jnp.int32))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits


def generate(model: nn.Module, params: FrozenVariableDict, idx: jnp.ndarray, max_new_tokens: int):
    # idx is (B, T) array of indices in the current context
    key = jax.random.PRNGKey(4223)

    for i in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

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
    return idx


def estimate_loss(data_dict, model: GPTLanguageModel, params, dropout_rng, eval_iters):
    def loss_fn(model: GPTLanguageModel, params, inputs, labels):
        logits = model.apply(params, inputs, rngs={"dropout": dropout_rng})
        B, T, C = logits.shape
        labels = labels.reshape(B * T)
        logits = logits.reshape(B * T, C)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)

    out = {}
    model.deterministic = True
    for split in ["train", "validation"]:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dict, block_size, batch_size, for_pmap=False)
            loss = loss_fn(model, params, X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.deterministic = False
    return out


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


def load_data(data_path: str) -> Tuple[str, GraphemeVocab]:
    csv.field_size_limit(MAX_FIELD_SIZE)
    with open(data_path, "r") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        stories = [row["Text"] for row in reader]
    vocab = GraphemeVocab()
    vocab.build_vocab(stories)
    return " ".join(stories), vocab


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
    encode = lambda s: [stoi[c] for c in Shabdansh(s)]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    return encode, decode


def get_model_n_params(
    vocab_size: int,
    n_embd: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    dropout: float,
    data_dict: Dict[str, torch.tensor],
    batch_size: int,
    for_validation: bool = False,
) -> Union[Tuple[GPTLanguageModel, FrozenVariableDict, Any], GPTLanguageModel]:
    # prepare rngs
    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}

    # disable dropout when for validation
    deterministic = for_validation

    # create model
    gpt = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout, deterministic)

    if not for_validation:
        # get params
        x, _ = get_batch("train", data_dict, block_size, batch_size)
        params = gpt.init(rngs, x)

        return gpt, params, dropout_rng

    else:
        return gpt


# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 20
n_embd = 384
n_head = 6
n_layer = 12
dropout = 0.2

text, vocab = load_data("/home/khandelia1000/premchand/data/premchand.tsv")
vocab_size = len(vocab.stoi)

encode, decode = get_encoder_decoder(vocab)

data_dict = get_data_dict(text, encode)

initialise_tracking()
# Initialize the parameters of the module
gpt, params, dropout_rng = get_model_n_params(
    vocab_size,
    n_embd,
    block_size,
    n_layer,
    n_head,
    dropout,
    data_dict,
    batch_size,
)
max_new_tokens = 10

parameter_count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6
logger.info(f"Number of parameters (in millions): {parameter_count}")


state = TrainState.create(apply_fn=gpt.apply, params=params, tx=optax.adamw(learning_rate))
state = jax_utils.replicate(state)

p_update = jax.pmap(update, axis_name="batch")
for iter in range(max_iters):
    # sample a batch of data
    xb, yb = get_batch("train", data_dict, block_size, batch_size, for_pmap=True)

    # evaluate the loss
    loss, state = p_update(state, xb, yb)

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        logger.info(estimate_loss(data_dict, gpt, jax_utils.unreplicate(state).params, dropout_rng, eval_iters))
        logger.info(f"step {iter}: train loss {jnp.mean(loss):.4f}")
        context = jnp.zeros((1, 1), dtype=jnp.int32)
        logger.info(
            decode(
                generate(gpt, jax_utils.unreplicate(state).params, context, max_new_tokens=max_new_tokens)[0].tolist()
            )
        )
