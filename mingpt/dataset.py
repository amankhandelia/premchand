from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import jax
from einops import rearrange
from tokenizers import Tokenizer
from datasets import Dataset


class DesiDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["Body"]
        tokens = np.asarray(self.tokenizer.encode(text).ids, dtype=np.int32)
        return tokens


def collate_fn(batch, tokenizer, seq_len: int, pmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    batch_x = []
    batch_y = []

    for tokens in batch:
        for i in range(0, len(tokens), seq_len):
            chunk_x = tokens[i : i + seq_len]
            chunk_y = tokens[i + 1 : i + seq_len + 1]

            # chunk the input based seq_len
            if len(chunk_y) < seq_len:
                pad_len = seq_len - len(chunk_y)
                chunk_y = np.concatenate([chunk_y, np.full((pad_len,), tokenizer.pad_token_id, dtype=np.int64)], axis=0)

            if len(chunk_x) < seq_len:
                pad_len = seq_len - len(chunk_x)
                chunk_x = np.concatenate([chunk_x, np.full((pad_len,), tokenizer.pad_token_id, dtype=np.int64)], axis=0)

            # stack all the inputs into one
            batch_x.append(chunk_x)
            batch_y.append(chunk_y)

    batch_x, batch_y = np.stack(batch_x), np.stack(batch_y)

    if pmap:
        device_count = jax.device_count()

        # if the batch count exceeds multiple of device count
        extra_batch_count = batch_x.shape[0] % device_count
        if extra_batch_count != 0:
            batch_x = batch_x[:-extra_batch_count]
            batch_y = batch_y[:-extra_batch_count]

        # reshape it for multiple devices
        batch_x = rearrange(
            batch_x, "(device_count batch_size) seq_len->device_count batch_size seq_len", device_count=device_count
        )
        batch_y = rearrange(
            batch_y, "(device_count batch_size) seq_len->device_count batch_size seq_len", device_count=device_count
        )

    return batch_x, batch_y
