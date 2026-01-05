# transformer

Transformerをゼロから実装して文章生成。GPT等のLLMの基礎を理解する。

## 仕組み

RNN（順番に処理）ではなく、Self-Attentionで「全体を一度に見る」。

```
入力: 「私 は 寿司 が」
        ↓
   [Token Embedding] 文字→ベクトル
        ↓
   [Positional Encoding] 位置情報を追加
        ↓
   [Transformer Block] × N層
   ├── Self-Attention: 全単語の関係を計算
   └── Feed Forward: 変換
        ↓
   [Output Layer]
        ↓
出力: 「好き」の確率
```

## Google Colab で実行（推奨）

GPUを使って高速に学習できます。

1. [Google Colab](https://colab.research.google.com/) を開く
2. `transformer_colab.ipynb` をアップロード
3. ランタイム → ランタイムのタイプを変更 → **GPU** を選択
4. 上から順にセルを実行

## ローカルで実行

```bash
# セットアップ
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 学習
make train TEXT=data/input.txt EPOCHS=10

# 保存済みモデルで生成
make generate START="The "
```

## ファイル構成

```
transformer/
├── transformer_colab.ipynb  # Colab用ノートブック
├── Makefile
└── transformer/
    ├── attention.py   # Self-Attention (Q, K, V)
    ├── model.py       # Transformer本体
    ├── data.py        # データセット
    ├── train.py       # 学習ループ
    └── generate.py    # 文章生成
```

## RNNとの比較

| | RNN (LSTM) | Transformer |
|--|--|--|
| 処理方法 | 順番に1文字ずつ | 全文字を同時に |
| 長距離の関係 | 苦手（忘れる） | 得意（Attention） |
| 並列化 | 不可 | 可能（GPU向き） |
| 位置情報 | 順番で暗黙的 | Positional Encodingで明示 |

## ロードマップ

- [x] Level 1: Self-Attention の実装
- [x] Level 2: Transformer ブロック
- [x] Level 3: Positional Encoding
- [x] Level 4: 学習ループ
- [x] Level 5: 文章生成
- [ ] Level 6: より大きなモデルで実験
