# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Machine learning study repository containing multiple projects. Each project has its own README.md and CLAUDE.md with specific instructions.

## Projects

- **dog-cat-classifier/**: Image classification with PyTorch (pretrained models → custom CNN)
- **text-generator/**: Character-level RNN for text generation
- **transformer/**: Transformer from scratch for text generation

## Common Setup

Each project uses Python venv. Setup per project:

```bash
cd <project-dir>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
