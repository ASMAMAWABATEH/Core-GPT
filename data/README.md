# Data

Place the raw dataset file at:

`data/raw/tiny_shakespeare.txt`

Then preprocess it with:

```bash
python scripts/preprocess_data.py
```

This will create `data/processed/tiny_shakespeare.pt`, which contains the encoded
text tensor and vocabulary used during training.

Note: Make sure you have the right to use and redistribute any dataset you place
in `data/raw/`.
