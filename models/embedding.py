import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=len(word2idx),
    embedding_dim=256
)