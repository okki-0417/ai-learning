import torch

from .model import Transformer
from .data import TextDataset


def generate(
    model: Transformer,
    dataset: TextDataset,
    start_text: str = "The ",
    length: int = 200,
    temperature: float = 0.8,
) -> str:
    device = next(model.parameters()).device
    model.eval()

    # 開始テキストをエンコード
    tokens = dataset.encode(start_text).unsqueeze(0).to(device)
    generated = start_text

    with torch.no_grad():
        for _ in range(length):
            # 因果マスク
            mask = Transformer.create_causal_mask(tokens.size(1), device)

            # 予測
            logits = model(tokens, mask)

            # 最後の位置の予測を取得
            next_logits = logits[0, -1, :] / temperature

            # サンプリング
            probs = torch.softmax(next_logits, dim=0)
            next_token = torch.multinomial(probs, 1)

            # 追加
            next_char = dataset.idx_to_char[next_token.item()]
            generated += next_char

            # 次の入力
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            # 長すぎる場合は切り詰め
            if tokens.size(1) > 512:
                tokens = tokens[:, -512:]

    return generated
