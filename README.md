# ai-learning

機械学習・深層学習を基礎から学ぶためのプロジェクト集。

## プロジェクト

### [dog-cat-classifier](./dog-cat-classifier/)
PyTorchを使った画像分類。事前学習済みモデルの利用から始めて、自作CNNまでレベルアップしていく。

| Level | 内容 | 状態 |
|-------|------|------|
| 1 | 事前学習済みモデル (ResNet18) | 完了 |
| 2 | 転移学習 | - |
| 3 | 犬種・猫種の細分類 | - |
| 4 | CNNをゼロから作成 | - |
| 5 | 自作CNNの精度改善 | - |
| 6 | Vision Transformer | - |

### [text-generator](./text-generator/)
文字レベルRNNによるテキスト生成。LLMの仕組みを理解するための第一歩。

| Level | 内容 | 状態 |
|-------|------|------|
| 1 | 文字レベルRNN (LSTM) | 完了 |
| 2 | 単語レベルRNN | - |
| 3 | Attention機構の追加 | - |
| 4 | 簡易Transformer | - |

### [transformer](./transformer/)
Transformerをゼロから実装して文章生成。GPT等のLLMの基礎を理解する。

| Level | 内容 | 状態 |
|-------|------|------|
| 1 | Self-Attention の実装 | 完了 |
| 2 | Transformer ブロック | 完了 |
| 3 | Positional Encoding | 完了 |
| 4 | 学習ループ | 完了 |
| 5 | 文章生成 | 完了 |
| 6 | より大きなモデルで実験 | - |

## 技術スタック

- Python 3
- PyTorch

## セットアップ

各プロジェクトディレクトリで:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 目標

ライブラリを「使う」だけでなく、中身を理解して「作れる」ようになること。
