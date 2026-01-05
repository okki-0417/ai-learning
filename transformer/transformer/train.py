import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import Transformer
from .data import TextDataset


def train(
    text: str,
    epochs: int = 10,
    batch_size: int = 32,
    seq_length: int = 128,
    embed_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    lr: float = 3e-4,
    resume_from: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # データ準備
    dataset = TextDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"語彙数: {dataset.vocab_size}文字")
    print(f"データ長: {len(dataset.data)}文字")
    print(f"バッチ数: {len(dataloader)}/エポック")

    # モデル作成
    if resume_from:
        print(f"モデル読み込み: {resume_from}")
        checkpoint = torch.load(resume_from, weights_only=False)
        model = Transformer(
            dataset.vocab_size,
            checkpoint["embed_size"],
            checkpoint["num_heads"],
            checkpoint["num_layers"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = Transformer(
            dataset.vocab_size,
            embed_size,
            num_heads,
            num_layers,
        ).to(device)

    # パラメータ数を表示
    num_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 学習ループ
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"\r  Epoch {epoch+1}: {batch_idx}/{len(dataloader)}", end="", flush=True)

            x, y = x.to(device), y.to(device)

            # 因果マスク
            mask = Transformer.create_causal_mask(x.size(1), device)

            # 順伝播
            logits = model(x, mask)

            # 損失計算
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}      ")

    # モデルを保存
    save_path = "model.pth"
    torch.save({
        "model_state": model.state_dict(),
        "chars": dataset.chars,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
    }, save_path)
    print(f"モデルを保存: {save_path}")

    return model, dataset
