import torch
import torch.nn as nn
import math


# ==========================================================
# Token Embedding
# ==========================================================

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        return self.embedding(x)


# ==========================================================
# Positional Encoding
# ==========================================================

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=1000):
        super().__init__()

        self.pe = torch.zeros(
            1,
            max_len,
            embed_dim
        )

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        self.pe[0,:,0::2] = torch.sin(
            position * div_term
        )

        self.pe[0,:,1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pos_embedding",
            self.pe
        )


    def forward(self,x):

        seq_len = x.size(1)

        x = x + self.pos_embedding[:, :seq_len]

        return x

# ==========================================================
# Multi-Head Self Attention
# ==========================================================

class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            V
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.embed_dim
        )

        output = self.fc_out(output)

        return output


# ==========================================================
# Feed Forward Network
# ==========================================================

class FeedForward(nn.Module):

    def __init__(self, embed_dim, expansion=4):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                expansion * embed_dim
            ),

            nn.ReLU(),

            nn.Linear(
                expansion * embed_dim,
                embed_dim
            )

        )

    def forward(self, x):

        return self.network(x)


# ==========================================================
# Decoder Block
# ==========================================================

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        self.attention = SelfAttention(
            embed_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        attention = self.attention(x)

        x = self.norm1(
            x + attention
        )

        forward = self.ffn(x)

        x = self.norm2(
            x + forward
        )

        return x


# ==========================================================
# GPT Model
# ==========================================================

class GPT(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_dim,
            num_heads,
            num_layers,
            max_length=1000
    ):

        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            embed_dim
        )

        self.position = PositionalEncoding(
            embed_dim,
            max_length
        )

        self.layers = nn.ModuleList(

            [
                DecoderBlock(
                    embed_dim,
                    num_heads
                )

                for _ in range(num_layers)
            ]

        )

        self.fc_out = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.position(x)

        for layer in self.layers:

            x = layer(x)

        x = self.fc_out(x)

        return x


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    model = GPT(

        vocab_size=10000,

        embed_dim=256,

        num_heads=4,

        num_layers=4,

        max_length=100

    )

    x = torch.randint(
        0,
        10000,
        (2, 20)
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)