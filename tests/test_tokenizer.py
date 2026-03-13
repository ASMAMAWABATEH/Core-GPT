import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from scratchgpt.tokenizer.tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    text = "hello"
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
