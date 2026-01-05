import argparse
from pathlib import Path

import torch

from .model import Transformer
from .data import TextDataset
from .train import train
from .generate import generate


def load_model(model_path: str):
    """保存済みモデルを読み込む"""
    checkpoint = torch.load(model_path, weights_only=False)
    chars = checkpoint["chars"]

    dataset = TextDataset.__new__(TextDataset)
    dataset.chars = chars
    dataset.vocab_size = len(chars)
    dataset.char_to_idx = {ch: i for i, ch in enumerate(chars)}
    dataset.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    model = Transformer(
        dataset.vocab_size,
        checkpoint["embed_size"],
        checkpoint["num_heads"],
        checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])

    return model, dataset


def main():
    parser = argparse.ArgumentParser(description="Transformerで文章生成")
    parser.add_argument("text_file", nargs="?", help="学習用テキストファイル")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start", type=str, default="The ", help="生成開始文字列")
    parser.add_argument("--length", type=int, default=500, help="生成する文字数")
    parser.add_argument("--load", type=str, help="保存済みモデルを読み込んで生成のみ")
    parser.add_argument("--resume", type=str, help="保存済みモデルから追加学習")
    args = parser.parse_args()

    if args.load:
        print(f"モデル読み込み: {args.load}")
        model, dataset = load_model(args.load)
    else:
        if not args.text_file:
            parser.error("text_file か --load が必要です")
        text = Path(args.text_file).read_text(encoding="utf-8")
        print(f"学習データ: {len(text)}文字")
        print("\n=== 学習開始 ===")
        model, dataset = train(text, epochs=args.epochs, resume_from=args.resume)

    print("\n=== 文章生成 ===")
    generated = generate(model, dataset, start_text=args.start, length=args.length)
    print(generated)


if __name__ == "__main__":
    main()
