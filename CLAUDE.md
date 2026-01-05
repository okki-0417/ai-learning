# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dog vs cat image classifier using PyTorch. Goal is to learn ML fundamentals by progressively leveling up the implementation:

1. Pretrained model (current) → 2. Transfer learning → 3. Breed classification → 4. CNN from scratch → 5. Advanced architectures

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run classification
python -m dog_cat_classifier <image_path>

# Inspect model weights (run after classification to ensure weights are cached)
python -m dog_cat_classifier.inspect_model
```

## Architecture

The `dog_cat_classifier` package follows a pipeline: `model.py` → `preprocess.py` → `classifier.py` → `presenter.py`

- **model.py**: Loads pretrained ResNet18 with ImageNet weights. Defines dog class IDs (151-268) and cat class IDs (281-285).
- **preprocess.py**: Image preprocessing with ImageNet normalization (resize to 256, center crop to 224).
- **classifier.py**: Runs inference and sums probabilities across dog/cat class ranges to get final prediction.
- **schemas.py**: Dataclasses for `ClassificationResult`, `Prediction`, and `TopNResult`.
- **presenter.py**: Formats and prints classification results in Japanese.
- **inspect_model.py**: Utility to examine ResNet18 weight structure.

Model weights cached at `~/.cache/torch/hub/checkpoints/`
