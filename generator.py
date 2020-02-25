import torch
import numpy
import random

import numpy as np

from scipy.special import softmax
from vocab import HindiVocab
from network import PremchandLanguageModel, PremchandTransformerLM



class TextGenerator:
    def __init__(self, model, vocab, strategy='argmax', top_n=1):
        """
        @param distribution[Sequence[float]]: A probability distribution over the set of token from which we wish to sample
        @param strategy[str]: Method which will operate on the distribution to sample the token
        """
        self.model = model
        self.vocab = vocab
        self.strategy = strategy
        self.top_n = top_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def sample_token(self, distribution):
        """
        Sample a token based on the distribution and the strategy opted

        @return token[int]: Token sampled
        """
        # decoding strategies picked up from the following
        # https://youtu.be/4uG1NMKNWCU?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&t=1014

        if self.strategy is 'argmax':
            token = np.argmax(distribution)
        elif self.strategy is 'pure_sample':
            token = np.random.choice(numpy.arange(0, len(distribution)), p=distribution)
        elif self.strategy is 'top_n':
            top_n_dist = sorted(zip(range(0, len(distribution)), distribution), reverse=True, key=lambda x: x[1])[:self.top_n]
            distribution = softmax([item[1] for item in top_n_dist])
            choices = [item[0] for item in top_n_dist]
            token = np.random.choice(choices, p=distribution)
        return int(token)

    def generate_text_from_transformers(self, generate_n, prime_text=None):
        sampled_tokens = []
        with torch.no_grad():
            if prime_text is not None:
                original = prime_text
                tokens = self.vocab.encode(original)
            else:
                token = random.randint(0, self.vocab.vocab_size - 1)
                tokens = [token]
                original = self.vocab.decode(tokens)
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            for i in range(generate_n):
                probs = self.model(tokens.unsqueeze(0))
                token = self.sample_token(probs.squeeze(dim=0)[-1].numpy())
                sampled_tokens.append(token)
                tokens = torch.cat([tokens, torch.tensor([token], dtype=torch.long, device=self.device)])
                print(self.vocab.decode(tokens))

        generated_text = self.vocab.decode(sampled_tokens)
        return original, generated_text

    def generate_text(self, generate_n, prime_text=None, temperature=0.8):
        """
        Generate text based on the prime_text passed in.

        @param generate_n[int]: Number of tokens to be generated
        @param prime_text[str]: String for priming the model
        @param temperature[float]: For treating softmax

        @returns (original[str], generated_text[str]): A tuple of text, first one being the original text and other generated by the model based on the original text
        """
        original = ""
        state = (torch.zeros(1, 1, self.model.hidden_dim, dtype=torch.float),
                 torch.zeros(1, 1, self.model.hidden_dim, dtype=torch.float))
        sampled_tokens = []

        with torch.no_grad():
            if prime_text is not None:
                original = prime_text
                tokens = self.vocab.encode(original)
                for token in tokens[:-1]:
                    state = self.model.get_next_state(token, state)
                token = tokens[-1]
            else:
                token = random.randint(0, self.vocab.vocab_size - 1)
                original = self.vocab.decode([token])

            for i in range(generate_n):
                probs, state = self.model.get_next(token, state)
                token = self.sample_token(probs.numpy())
                sampled_tokens.append(token)

        generated_text = self.vocab.decode(sampled_tokens)
        return original, generated_text

    @classmethod
    def get_generator_lstm(cls, model_path, vocab_size, emb_dim, hidden_dim, strategy='argmax', top_n=1):
        model = PremchandLanguageModel.load_model(model_path, vocab_size, emb_dim, hidden_dim)
        vocab = HindiVocab.from_bpemb(vocab_size=vocab_size)
        return cls(model, vocab, strategy=strategy, top_n=top_n)

    @classmethod
    def get_generator_transformer(cls, model_path, d_model, d_ff, num_heads, vocab_size, num_layers, strategy='argmax', top_n=1):
        model = PremchandTransformerLM.load_model(model_path, d_model, d_ff, num_heads, vocab_size, num_layers)
        vocab = HindiVocab.from_bpemb(vocab_size=vocab_size)
        return cls(model, vocab, strategy=strategy, top_n=top_n)

if __name__ == '__main__':
    model_path = '/home/aman/Downloads/model_0074.pth'
    num_layers = 2
    d_model = 256
    d_ff = 256
    num_heads = 4
    vocab_size = 10000
    generator = TextGenerator.get_generator_transformer(model_path, d_model, d_ff, num_heads, vocab_size, num_layers)
    text = 'जब गाँव के सारे आदमी गाँव'
    next_n = 10
    print(generator.generate_text_from_transformers(next_n, text))