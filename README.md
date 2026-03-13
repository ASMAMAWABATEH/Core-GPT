# Core-GPT

![CI](https://github.com/ASMAMAWABATEH/Core-GPT/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Core-GPT is a research-grade, from-scratch implementation of a GPT-style decoder-only Transformer in PyTorch. It is designed for clarity, correctness, and experimentation. The project uses only low-level PyTorch primitives and standard Python libraries.

## Features
- BPE tokenizer (byte-pair encoding)
- Causal masked multi-head self-attention
- Decoder-only Transformer blocks
- Training pipeline with checkpoints, logging, and reproducibility
- Autoregressive text generation with greedy, temperature, top-k, and top-p sampling
- Modular codebase for easy extension

## Repository Layout
```
Core-GPT/
├── README.md
├── requirements.txt
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── datasets/
│   ├── text_dataset.py
│   └── preprocessing.py
├── tokenizer/
│   ├── tokenizer.py
│   ├── bpe.py
│   └── vocab.py
├── models/
│   ├── gpt.py
│   ├── transformer_block.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── embedding.py
│   └── positional_encoding.py
├── training/
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── loss.py
├── inference/
│   ├── generate.py
│   └── sampling.py
├── utils/
│   ├── logger.py
│   ├── checkpoint.py
│   ├── seed.py
│   └── metrics.py
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
└── tests/
    ├── test_attention.py
    ├── test_transformer.py
    └── test_tokenizer.py
```

## Setup
Install dependencies:
```
pip install -r requirements.txt
```

## Data Preprocessing
The dataset is expected at `data/raw/tiny_shakespeare.txt`.

```
python scripts/preprocess_data.py
```

This will create `data/processed/tiny_shakespeare.pt` which contains the encoded text tensor and tokenizer metadata.
You can switch tokenizers if needed, e.g. `--tokenizer char` or adjust merges via `--bpe_merges`.

## Training
```
python scripts/train.py
```

Resume from a checkpoint:
```
python scripts/train.py --resume checkpoints/step_1000.pt
```

## Text Generation
```
python scripts/generate.py --checkpoint checkpoints/step_1000.pt --prompt "To be, or not to be"
```

## Example Output
After a short training run on Tiny Shakespeare, you can expect short but coherent sequences such as:
```
To be, or not to be, the king will say,
That I have seen the day to find my hand.
```

## Notes
- This project is intentionally small and educational.
- Increase model size and training steps for better output quality.

## Tests
```
pytest -q
```
