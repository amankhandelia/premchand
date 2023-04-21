import jax
import jax.numpy as jnp

from jax import lax

from flax.core.scope import FrozenVariableDict

from mingpt.gpt import GPTLanguageModel
from mingpt.config import ModelConfig


def generate(
    idx: jnp.ndarray,
    model: GPTLanguageModel,
    params: FrozenVariableDict,
    config: ModelConfig,
    max_new_tokens: int,
    dropout_rng,
    pad_id: int = 0,
):
    model.deterministic = True
    key = jax.random.PRNGKey(4223)

    # Pad the input idx to have the final shape
    prompt_token_count = idx.shape[-1]
    idx = jnp.pad(
        idx, ((0, 0), (0, config.arch.block_size - prompt_token_count)), mode="constant", constant_values=pad_id
    )

    def cond_fn(loop_state):
        _, _, counter = loop_state
        return counter < max_new_tokens

    def body_fn(loop_state):
        idx, key, counter = loop_state

        logits = model.apply(params, idx, rngs={"dropout": dropout_rng})
        # focus only on the counter step
        logits = logits[:, prompt_token_count + counter, :]  # becomes (B, C)

        # apply softmax to get probabilities
        probs = jax.nn.softmax(logits, axis=-1)  # (B, C)

        # sample from the distribution
        sample_key = jax.random.fold_in(key, counter)
        idx_next = jax.random.choice(sample_key, probs.shape[-1], shape=(1, 1), p=jnp.reshape(probs, (-1)))

        # append sampled index to the running sequence
        idx = lax.dynamic_update_slice(idx, idx_next, (0, prompt_token_count + counter))
        return idx, key, counter + 1

    idx, _, _ = lax.while_loop(cond_fn, body_fn, (idx, key, 0))

    model.deterministic = False
    return idx
