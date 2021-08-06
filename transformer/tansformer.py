# 해당 코드는 아래의 Reference를 참고해서 만들었습니다.
# http://incredible.ai/nlp/2020/02/29/Transformer/

import math
import torch
import torch.nn as nn
from torch.nn.modules import dropout

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
        self.sqrt_dk = d_k**0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self,):
        