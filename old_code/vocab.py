from bpemb import BPEmb

class HindiVocab:
    def __init__(self, tokenizer, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def encode(self, text):
        """
        Encode the text into an array of token ids

        @param text[str]: Text to be encoded

        @returns [Sequence[int]]: An array of token ids representing the text passed
        """
        return self.tokenizer.encode_ids(text)

    def decode(self, tokens):
        """
        Decode an array of token ids to text it originally represents

        @param tokens[Sequence[int]]: A list of tokens, each token a integer

        @return [str]: Textual representation of the tokens passed
        """
        return self.tokenizer.decode_ids(tokens)

    @classmethod
    def from_bpemb(cls, vocab_size):
        """
        Get a tokenizer based on BPE (Byte Pair Encoding)

        @param vocab_size[int]: Size of the vocab based on which tokenizer will be initialized

        @return vocab[HindiVocab]: Instance of the class
        """
        tokenizer = BPEmb(lang="hi", dim=25, vs=vocab_size)

        return cls(tokenizer, vocab_size)
