"""
Utilities for dealing with embeddings.
"""
import torch.nn as nn
import torch
import numpy as np


def get_embeddings(vocab, word2vec, special_words):
    N = len(vocab['question_token_to_idx'])
    padding = vocab['question_token_to_idx']['<NULL>']
    D = word2vec.vector_size
    embed = nn.Embedding(N, D, padding_idx=padding)
    # print(type(embed.weight))
    for word, idx in vocab['question_token_to_idx'].items():
        if word == '<NULL>': continue
        if word in special_words:
            embed.weight.data[idx] = torch.randn(D)
        else:
            embed.weight.data[idx] = torch.from_numpy(word2vec[word])
    return embed


def expand_embedding_vocab(embed, token_to_idx, word2vec=None, std=0.01):
    old_weight = embed.weight.data
    old_N, D = old_weight.size()
    new_N = 1 + max(idx for idx in token_to_idx.values())
    new_weight = old_weight.new(new_N, D).normal_().mul_(std)
    new_weight[:old_N].copy_(old_weight)
    
    if word2vec is not None:
        num_found = 0
        assert D == word2vec['vecs'].size(1), 'Word vector dimension mismatch'
        word2vec_token_to_idx = {w: i for i, w in enumerate(word2vec['words'])}
        for token, idx in token_to_idx.items():
            word2vec_idx = word2vec_token_to_idx.get(token, None)
            if idx >= old_N and word2vec_idx is not None:
                vec = word2vec['vecs'][word2vec_idx]
                new_weight[idx].copy_(vec)
                num_found += 1
    embed.num_embeddings = new_N
    embed.weight.data = new_weight
    return embed
