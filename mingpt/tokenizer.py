import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
from histr import Shabdansh

from tokenizers import Tokenizer, decoders, models, Regex, pre_tokenizers, trainers


def is_devanagari(text):
    """
    Returns True if the input string only contains Devanagari characters, False otherwise.
    """
    for char in text:
        if not 0x0900 <= ord(char) <= 0x097F:
            return False
    return True


def get_clusters(dataset: Dataset):
    # Collect all the unique strings
    all_clusters = set()
    for row in tqdm(dataset):
        tokens = set(Shabdansh(row["Body"]).str_ls)
        tokens = {token for token in tokens if is_devanagari(token)}
        all_clusters.update(tokens)

    clusters = [grapheme for grapheme in all_clusters if Shabdansh.is_valid_cluster(grapheme)]

    return clusters


def batch_iterator(dataset: Dataset, batch_size: int):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]["Body"]
        yield batch


dataset = load_dataset("desiai/samachaar", split="train")


# read clusters.json
with open("notebooks/clusters.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)

# clusters = get_clusters(dataset)

pt = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
assert all([len(pt.pre_tokenize_str(cluster)) == 1 for cluster in clusters])
clusters_encoded = [pt.pre_tokenize_str(cluster)[0][0] for cluster in clusters]

split_regex = Regex(r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\u0900-\u097F]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


# build tokenizer
tokenizer = Tokenizer(models.Unigram())
tokenizer.add_tokens(clusters_encoded)
# add multiple pretokenizer to the tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(split_regex, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False),
    ]
)
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, use_regex=False)


trainer = trainers.UnigramTrainer(
    vocab_size=32768 - (len(clusters_encoded) + 256 + 6),
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    show_progress=True,
)

# train tokenizer
batch_size = 1000000
tokenizer.train_from_iterator(batch_iterator(dataset, batch_size), trainer=trainer, length=len(dataset))

tokenizer.save("devnagari_test_tokenizer.json")
