from typing import Optional

from flax import linen as nn
import jax.numpy as jnp

from memory_efficient_attention import efficient_dot_product_attention_jax


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    num_heads: int
    head_size: int
    n_embd: int
    dropout_rate: float
    block_size: int
    deterministic: Optional[bool] = None

    def setup(self):
        self.key_projection = nn.Dense(self.n_embd, use_bias=False)
        self.query_projection = nn.Dense(self.n_embd, use_bias=False)
        self.value_projection = nn.Dense(self.n_embd, use_bias=False)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.tril = jnp.tril(jnp.ones((self.block_size, self.block_size)))
        self.proj = nn.Dense(self.n_embd, use_bias=True)
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def _split_heads(self, x):
        return x.reshape(x.shape[:2] + (self.num_heads, self.head_size))

    def _merge_heads(self, x):
        return x.reshape(x.shape[:2] + (self.n_embd,))

    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, n_embd)
        batch_size = x.shape[0]

        key = self.key_projection(x)
        query = self.query_projection(x)
        value = self.value_projection(x)

        key = self._split_heads(key)
        query = self._split_heads(query)
        value = self._split_heads(value)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Repeat the array b times along the first dimension
        mask = jnp.repeat(self.tril[jnp.newaxis, :, :], batch_size * self.num_heads, axis=0)

        # Reshape the array to shape (b, h, x, x)
        mask = mask.reshape(batch_size, self.num_heads, self.block_size, self.block_size)
        bias = jnp.zeros(mask.shape, dtype=jnp.float32)

        heads_out = efficient_dot_product_attention_jax(query, key, value, mask=mask, bias=bias)
        heads_out = self._merge_heads(heads_out)
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
