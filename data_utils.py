import jax
import jax.numpy as jnp

from einops import rearrange

from flax.core.scope import FrozenVariableDict

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
