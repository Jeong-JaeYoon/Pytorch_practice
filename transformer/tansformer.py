# 해당 코드는 아래의 Reference를 참고해서 만들었습니다.
# http://incredible.ai/nlp/2020/02/29/Transformer/

import math
import torch
import torch.nn as nn

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
    