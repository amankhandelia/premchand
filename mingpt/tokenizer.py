import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
from histr import Shabdansh

from tokenizers import Tokenizer, decoders, models, Regex, pre_tokenizers, trainers
from collections import Counter


def is_devanagari(text):
    """
    Returns True if the input string only contains Devanagari characters, False otherwise.
    """
    for char in text:
        if not 0x0900 <= ord(char) <= 0x097F:
            return False
    return True


def count_clusters(chunk):
    # get count of all clusters
    all_clusters = Counter()
    for text in chunk["Body"]:
        all_clusters.update(text)

    # list of empty tuples to maintain the same count as map
    counters = [[()] for _ in range(len(chunk["Body"]) - 1)]

    # convert it into a list, as python objects or dict are not supported in hf dataset
    all_clusters = [(key, str(value)) for key, value in all_clusters.items()]
    counters.append(all_clusters)

    # create a mark column to filter out the actual counters
    mark = [0] * (len(chunk["Body"]) - 1)
    mark.append(1)

    return {"counters": counters, "mark": mark}


def process_text(row):
    valid_clusters = [
        cluster
        for cluster in set(Shabdansh(row["Body"]).str_ls)
        if is_devanagari(cluster) and Shabdansh.is_valid_cluster(cluster)
    ]
    return {"Body": valid_clusters}


def get_clusters(dataset: Dataset):
    # convert text to list of clusters
    dataset = dataset.map(process_text, num_proc=32)

    # batch text in large group and count clusters
    dataset = dataset.map(count_clusters, num_proc=32, batched=True, batch_size=10000)

    # filter out the rows which contain the actual count values
    mark_column = dataset["mark"]
    counter_indices = [idx for idx, mark in enumerate(mark_column) if mark == 1]
    counters = dataset.select(counter_indices)

    # accumulate all counters into one
    all_clusters = Counter()
    for row in tqdm(counters):
        counter = Counter({key: int(value) for key, value in row["counters"]})
        all_clusters.update(counter)

    return all_clusters


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
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ]
)
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=False)


trainer = trainers.UnigramTrainer(
    vocab_size=32768 - (len(clusters_encoded) + 256 + 6),
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    show_progress=True,
)

# train tokenizer
batch_size = 1000000
tokenizer.train_from_iterator(batch_iterator(dataset, batch_size), trainer=trainer, length=len(dataset))

tokenizer.save("devnagari_tokenizer.json")
