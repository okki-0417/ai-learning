"""
Self-Attention: Transformerの核心

Q (Query):  「何を探しているか」
K (Key):    「何を持っているか」
V (Value):  「実際の情報」

Attention(Q, K, V) = softmax(QK^T / √d) × V
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Single-Head Self-Attention

    入力の各位置が、他の全位置との関連度を計算して情報を集める
    """

    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

        # Q, K, V を作る線形変換
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_size)
        mask: 未来の情報を見せないためのマスク（文章生成時に使用）
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V を計算
        Q = self.query(x)  # (batch, seq_len, embed_size)
        K = self.key(x)
        V = self.value(x)

        # Attention スコア: QとKの類似度
        # (batch, seq_len, embed_size) × (batch, embed_size, seq_len)
        # → (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # スケーリング（勾配の安定化）
        scores = scores / math.sqrt(self.embed_size)

        # マスク適用（未来を見せない）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmaxで確率に変換
        attention_weights = torch.softmax(scores, dim=-1)

        # Valueを重み付け和
        output = torch.matmul(attention_weights, V)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    複数の視点（head）からAttentionを計算して統合
    例: head1は「主語-動詞」関係、head2は「形容詞-名詞」関係を学習
    """

    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V を計算して、headごとに分割
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # (batch, num_heads, seq_len, head_dim) に転置
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # headを結合
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)

        return self.out(output)
