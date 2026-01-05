import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """テキストを学習用データに変換する（text-generatorと同じ）"""

    def __init__(self, text: str, seq_length: int = 128):
        self.seq_length = seq_length

        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor([self.char_to_idx[ch] for ch in text])

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[ch] for ch in text])

    def decode(self, indices: torch.Tensor) -> str:
        return "".join(self.idx_to_char[int(i.item())] for i in indices)
