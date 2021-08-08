# 해당 코드는 아래의 Reference를 참고해서 만들었습니다.
# 부족한 코드
# http://incredible.ai/nlp/2020/02/29/Transformer/
# https://hongl.tistory.com/194
# https://github.com/hyunwoongko/transformer

import math
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.modules.normalization import LayerNorm

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(EmbeddingLayer,self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_seq_len = 400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,embed_dim,2).float()*(-math.log(10000.0)/embed_dim))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

def create_mask(src: torch.Tensor, trg: torch.Tensor, src_pad_idx: int, trg_pad_idx: int):
    src_mask = _create_padding_mask(src, src_pad_idx)
    trg_mask = None
    if trg is not None:
        trg_mask = _create_padding_mask(trg, trg_pad_idx)
        nopeak_mask = _create_nopeak_mask(trg)
        trg_mask = trg_mask & nopeak_mask

    return src_mask, trg_mask

def _create_padding_mask(seq: torch.Tensor, pad_idx: int):
    return (seq != pad_idx).unsqueeze(-2)

def _create_nopeak_mask(trg):
    batch_size, seq_len = trg.size()
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device = trg.device), diagonal=1)).bool()
    return nopeak_mask

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim, n_head, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.dk = embed_dim // n_head
        self.dv = embed_dim // n_head

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias = False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias = False)
        self.linear_f = nn.Linear(embed_dim, embed_dim, bias = False)

        self.attention = ScaleDotProductAttention(self.dk, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        
        batch_size, n_head, dk, dv = q.size(0), self.n_head, self.dk, self.dv

        q = self.linear_q(q).view(batch_size, -1, n_head, dk)
        k = self.linear_k(k).view(batch_size, -1, n_head, dk)
        v = self.linear_v(v).view(batch_size, -1, n_head, dv)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = self.attention(q, k, v, mask)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        scores = self.linear_f(scores)
        return scores

class ScaleDotProductAttention(nn.Module):

    def __init__(self, d_k, dropout):        
        super(ScaleDotProductAttention, self).__init__()
        self.sqrt_dk = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask = None):
        attention = torch.matmul(q,k.transpose(-2, -1)) / self.sqrt_dk
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention = attention.masked_fill(~mask, -1e9)

        attention = self.dropout(nn.Softmax(attention, dim=-1))
        output = torch.matmul(attention, v)
        return output

class PositionWiseFeedForward(nn.Module):

    def __init__(self, embed_dim, d_ff = 2048, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim = embed_dim, n_head = n_head, dropout = dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(p = dropout)

        self.ff = PositionWiseFeedForward(embed_dim = embed_dim, d_ff = d_ff, dropout = dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(p = dropout)

    def forward(self, x, src_mask):

        _x = x
        x = self.attention(q = x, k = x, v = x, mask = src_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ff(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x

class Encoder(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, dropout, max_seq_len, d_ff, n_head, n_layers):
        super().__init__()
        self.embed = EmbeddingLayer(vocab_size = vocab_size, embed_dim = embed_dim)
        self.position = PositionalEncoding(embed_dim = embed_dim, dropout = dropout, max_seq_len = max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim = embed_dim, n_head = n_head, d_ff = d_ff, dropout = dropout) for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.embed(x)
        x = self.position(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class DecoderLayer(nn.Module):
    
    def __init__(self, embed_dim, n_head, dropout, d_ff):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(embed_dim = embed_dim, n_head = n_head, dropout = dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(p = dropout)

        self.enc_attention = MultiHeadAttention(embed_dim = embed_dim, n_head = n_head, dropout = dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(p = dropout)

        self.ff = PositionWiseFeedForward(embed_dim, d_ff = d_ff, dropout = dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(p = dropout)

    def forward(self, dec, enc, src_mask, trg_mask):
        
        _x = dec
        x = self.masked_attention(q = dec, k = dec, v = dec, mask = trg_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.enc_attention(q = x, k = enc, v = enc, mask = src_mask)
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        _x = x
        x = self.ff(x)
        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()