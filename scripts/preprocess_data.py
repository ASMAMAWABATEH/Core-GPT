from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.preprocessing import build_and_save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Tiny Shakespeare dataset.")
    parser.add_argument("--raw_path", type=str, default="data/raw/tiny_shakespeare.txt")
    parser.add_argument("--output_path", type=str, default="data/processed/tiny_shakespeare.pt")
    args = parser.parse_args()

    raw_path = (PROJECT_ROOT / args.raw_path).resolve()
    output_path = (PROJECT_ROOT / args.output_path).resolve()

    build_and_save_dataset(raw_path, output_path)
    print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    main()
