import torch
import csv

from torch.utils.data import Dataset
from vocab import HindiVocab

class LanguageModelDataset(Dataset):
    def __init__(self, data, vocab, seq_len):
        """
        @param data: A torch tensor representing the data
        @param vocab: Tokenizer
        @param seq_len:
        """
        super(LanguageModelDataset).__init__()
        self.data = data
        self.vocab = vocab
        self.seq_len = seq_len
        self.batchify(self.seq_len)

    def batchify(self, seq_len):
        """
        Process the data as per the seq_len specificed and batch them accordingly

        @param seq_len[int]: Sequence length of each individual example
        """
        self.seq_len = seq_len
        self.bs = self.data.size(0) // self.seq_len
        self.data = self.data[:(self.bs * self.seq_len)]
        self.data = self.data.reshape(-1, self.seq_len)

    def __getitem__(self, index):
        """
        Return the source and target tensor at given index

        @param index: Index at which the dataset

        @returns source, target
        """
        if index == self.bs - 1:
            source = self.data[index][:-1]
            target = self.data[index][1:]
        else:
            source = self.data[index]
            target = torch.cat([self.data[index][1:], self.data[index + 1][:1]])
        return source, target

    def __len__(self):
        return self.data.size(0) - 1

    def save_data_tensor(self, tensor_path):
        torch.save(self.data.reshape(-1), tensor_path)

    def load_data_tensor(self, tensor_path):
        self.data = torch.load(tensor_path)

    @classmethod
    def from_tensor(cls, tensor_path, seq_len=32, vocab_size=10000):
        vocab = HindiVocab.from_bpemb(vocab_size=vocab_size)
        data = torch.load(tensor_path)
        return cls(data, vocab, seq_len)

    @classmethod
    def from_tsv(cls, tsv, seq_len=32, vocab_size=10000):
        """
        Read the tsv and instance of the LanguageModelDataset class

        @param tsv[str]: Complete path at which the tsv reside

        @returns dataset[LanguageModelDataset]: Instance of the class encapsulating the data in the tsv
        """
        corpus = ""
        with open(tsv, 'r', newline='\n') as tsvfile:
            csv.field_size_limit(10000000)
            reader = csv.reader(tsvfile, delimiter="\t")
            for record in reader:
                text = record[-1]
                text = text.replace("\n", "")
                corpus += text
        vocab = HindiVocab.from_bpemb(vocab_size=vocab_size)
        data = torch.LongTensor(vocab.encode(corpus))
        return cls(data, vocab, seq_len)
