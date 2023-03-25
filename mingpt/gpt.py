from typing import Optional

from flax import linen as nn
import jax.numpy as jnp
from einops import rearrange


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

    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        key = self.key(x)  # (B,T,hs)
        query = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        key = rearrange(key, "batch_size seq_len dim -> batch_size dim seq_len")
        wei = jnp.matmul(query, key) * (1 / jnp.sqrt(self.head_size))  # (B, T, T)
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
