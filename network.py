import torch
import torch.nn as nn
import torch.nn.functional as F

class PremchandLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        super(PremchandLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.vocab_projection = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, source):
        """
        @param source[torch.Tensor]: A tensor of torch.int type, where each element represents index of a token.
                                    Shape of Tensor should be [batch_size, max_seq_len]

        @returns probs[torch.Tensor]: A tensor of probabilties
        """
        source = source.permute(1, 0)  # [max_seq_len, batch_size]
        src_emb = self.embeddings(source)  # [max_seq_len, batch_size, embedding_size]
        hidden_states, _ = self.lstm(src_emb)  # [max_seq_len, batch_size, hidden_size]
        output = self.vocab_projection(hidden_states)  # [max_seq_len, batch_size, vocab_size]
        probs = F.log_softmax(output, dim=-1).permute(1, 0, 2)  # [batch_size, max_seq_len, vocab_size]
        return probs

    def get_next_state(self, token, last_state=None):
        device = next(self.lstm.parameters()).device
        token = torch.tensor([[token]], dtype=torch.long, device=device)  # [1, 1]
        src_emb = self.embeddings(token)  # [1, 1, embedding_size]
        if last_state:
            _, (hidden_state, cell_state) = self.lstm(src_emb,
                                                      last_state)  # ([1, 1, hidden_size], [1, 1, hidden_size])
        else:
            _, (hidden_state, cell_state) = self.lstm(src_emb)  # ([1, 1, hidden_size], [1, 1, hidden_size])
        return hidden_state, cell_state

    def get_next(self, token, last_state=None):
        hidden_state, cell_state = self.get_next_state(token, last_state)  # ([1, 1, hidden_size], [1, 1, hidden_size])
        output = self.vocab_projection(hidden_state)  # [1, 1, vocab_size]
        probs = F.softmax(output, dim=-1).squeeze()  # [1, 1, vocab_size]
        return probs, (hidden_state, cell_state)

    @classmethod
    def load_model(cls, model_path, vocab_size, embedding_dim, hidden_dim):
        model = cls(vocab_size, embedding_dim, hidden_dim)
        model.load_state_dict(torch.load(model_path))
        return model