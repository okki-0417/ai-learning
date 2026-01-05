# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transformer implementation from scratch for text generation. Learning project to understand how modern LLMs (GPT, etc.) work.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train
python -m transformer <text_file> --epochs 10

# Generate
python -m transformer --load model.pth --start "The"
```

## Architecture

- **model.py**: Transformer model (Self-Attention, Feed Forward, Positional Encoding)
- **attention.py**: Self-Attention mechanism
- **data.py**: Text tokenization and dataset
- **train.py**: Training loop
- **generate.py**: Text generation
