import torch

from torch import nn

from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_v, masked=False):
        """

        @param d_model:
        @param num_heads:
        @param d_v:
        @param masked[boolean]: Whether to mask the input or not
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model//self.num_heads
        self.d_q = self.d_k
        self.d_v = d_v
        self.masked = masked
        self.key_head_weights = [nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(num_heads)]
        self.query_head_weights = [nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(num_heads)]
        self.value_head_weights = [nn.Linear(self.d_model, self.d_v, bias=False) for _ in range(num_heads)]
        self.w_o = nn.Linear(self.num_heads*self.d_v, self.d_model, bias=False)
        self.layer_norm = nn.LayerNorm(self.d_model)


    def forward(self, tokens):
        """
        @param tokens[torch.Tensor]: A tensor of torch.float type, where contains the embedding of the tokens
                                    Shape of Tensor should be [batch_size, max_seq_len, d_model]

        @return: A tensor of torch.float type of same shape as tokens
        """

        multi_heads = []
        for i in range(self.num_heads):
            q_i = self.query_head_weights[i](tokens) # [batch_size, max_seq_len, self.d_q]
            v_i = self.value_head_weights[i](tokens) # [batch_size, max_seq_len, self.d_v]
            k_i = self.key_head_weights[i](tokens) # [batch_size, max_seq_len, self.d_k]
            attn_score = self.attention(q_i, k_i) # [batch_size, max_seq_len, max_seq_len]
            if self.masked:
                mask = self.generate_masks(tokens.size(1)) # [max_seq_len, max_seq_len]
                attn_score.masked_fill_(mask == 0., -1e-9) # [batch_size, max_seq_len, max_seq_len]
            attn_score = F.softmax(attn_score, dim=-1) # [batch_size, max_seq_len, max_seq_len]
            head_i = torch.bmm(attn_score, v_i) # [batch_size, max_seq_len, self.d_k]
            multi_heads.append(head_i) # [batch_size, max_seq_len, self.d_k] * self.num_heads
        multi_head = torch.cat(multi_heads, dim=-1) # [batch_size, max_seq_len, self.d_v * self.num_heads]
        multi_head = self.w_o(multi_head) # [batch_size, max_seq_len, self.d_model]
        output = multi_head + tokens # [batch_size, max_seq_len, self.d_model]
        output = self.layer_norm(output) # [batch_size, max_seq_len, self.d_model]
        return output

    def generate_masks(self, max_seq_len):
        mask = torch.ones(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            mask[i, i + 1:] = 0.
        return mask


    def attention(self, query, key):
        """
        Generate attention score by attending over the key vectors using the query vectors

        Implementation of scaled dot-product attention as described in arXiv:1706.03762v5

        @param query[torch.Tensor]: A tensor of type torch.float with dimensions [batch_size, max_seq_len, self.d_q]
        @param key[torch.Tensor]: A tensor of type torch.float with dimensions [batch_size, max_seq_len, self.d_v]

        @return: A tensor of type torch.float with dimensions [batch_size, max_seq_len, max_seq_len]
        """
        key = key.permute(0, 2, 1)
        attn = torch.div(torch.bmm(query, key), torch.sqrt(torch.tensor(self.d_k, dtype=torch.float)))

        # permute as the attention vector for each query vector is in form of column vector
        attn = attn.permute(0, 2, 1)
        return attn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.w_2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, tokens):
        ffn = self.w_2(F.relu(self.w_1(tokens)))
        ffn += tokens
        output = self.layer_norm(ffn)
        return output

class LanguageModelTransformer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        """

        @param d_model:
        @param d_ff:
        @param num_heads:
        """
        super(LanguageModelTransformer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        if self.d_model//self.num_heads != self.d_model/self.num_heads:
            raise Exception("d_model should be a multiple of num_heads but was d_model: {}, num_heads: {}".format(self.d_model, self.num_heads))
        self.d_v = self.d_model // self.num_heads
        self.attention = MultiHeadAttention(self.d_model, self.num_heads, self.d_v, masked=True)
        self.ffn = FeedForwardNetwork(self.d_model, self.d_ff)

    def forward(self, tokens):
        out = self.attention(tokens)
        out = self.ffn(out)
        return out

