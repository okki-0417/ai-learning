"""
Transformer モデル

構成:
1. Token Embedding: 文字 → ベクトル
2. Positional Encoding: 位置情報を追加
3. Transformer Blocks × N
4. Output Layer: ベクトル → 次の文字の確率
"""

import torch
import torch.nn as nn
import math

from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    位置エンコーディング

    Transformerは順番を見ないので、位置情報を明示的に追加する
    sin/cos関数で各位置にユニークなパターンを付与
    """

    pe: torch.Tensor

    def __init__(self, embed_size: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数番目
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数番目

        pe = pe.unsqueeze(0)  # (1, max_len, embed_size)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class FeedForward(nn.Module):
    """
    Feed Forward Network

    Attention後の変換。各位置を独立に処理。
    """

    def __init__(self, embed_size: int, hidden_size: int | None = None):
        super().__init__()
        hidden_size = hidden_size or embed_size * 4

        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),  # 活性化関数（ReLUより滑らか）
            nn.Linear(hidden_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer ブロック

    Attention → Add & Norm → FeedForward → Add & Norm
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + 残差接続
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward + 残差接続
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class Transformer(nn.Module):
    """
    文章生成用 Transformer (GPT風)

    入力: 文字列のインデックス列
    出力: 次の文字の確率分布
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_size = embed_size

        # トークン埋め込み
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # 位置エンコーディング
        self.pos_encoding = PositionalEncoding(embed_size, max_len)

        # Transformerブロック × N
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 出力層
        self.ln_final = nn.LayerNorm(embed_size)
        self.output = nn.Linear(embed_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) トークンインデックス
        """
        # 埋め込み + 位置エンコーディング
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformerブロックを通す
        for block in self.blocks:
            x = block(x, mask)

        # 出力
        x = self.ln_final(x)
        logits = self.output(x)

        return logits

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device):
        """
        因果マスク: 未来のトークンを見せない

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
