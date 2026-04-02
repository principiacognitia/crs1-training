"""
Data preparation for CRS-1 backbone training.
Converts corpus-L{1-4}.jsonl → binary token arrays for nanoGPT.

Usage:
    python prepare_data.py --data_dir code/domain/ --out_dir backbone/data/ [--levels 1 2 3 4]
"""

import json
import argparse
import struct
from pathlib import Path
from tokenizer import MinicalcTokenizer


def load_corpus(data_dir: str, levels: list[int]) -> list[dict]:
    """Load JSONL corpus for given levels."""
    examples = []
    for lvl in levels:
        path = Path(data_dir) / f'corpus-L{lvl}.jsonl'
        if not path.exists():
            print(f'Warning: {path} not found, skipping')
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    ex['level'] = lvl
                    examples.append(ex)
    return examples


def encode_example(tok: MinicalcTokenizer, ex: dict) -> list[int]:
    """Encode input→output as a single sequence: <BOS> input <SEP> output <EOS>"""
    # Format: "<BOS>{input} = {output}<EOS>" or just the input expression
    text = ex.get('input', '')
    output = ex.get('output', '')
    label = ex.get('correct', True)

    # Build sequence: question + answer
    if output:
        full = f"{text} = {output}"
    else:
        full = text

    return tok.encode(full, add_special=True)


def prepare(data_dir: str, out_dir: str, levels: list[int], split: float = 0.9):
    """Tokenize corpus and save as binary uint16 arrays."""
    tok = MinicalcTokenizer()
    tok.save(Path(out_dir) / 'tokenizer.json')

    examples = load_corpus(data_dir, levels)
    print(f'Loaded {len(examples)} examples from levels {levels}')

    # Shuffle with fixed seed
    import random
    random.seed(42)
    random.shuffle(examples)

    n_train = int(len(examples) * split)
    splits = {
        'train': examples[:n_train],
        'val': examples[n_train:],
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits.items():
        all_ids = []
        for ex in split_examples:
            ids = encode_example(tok, ex)
            all_ids.extend(ids)

        # Save as binary uint16 (nanoGPT format)
        out_path = Path(out_dir) / f'{split_name}.bin'
        with open(out_path, 'wb') as f:
            # Header: vocab_size (uint32) + n_tokens (uint64)
            f.write(struct.pack('<I', tok.vocab_size))
            f.write(struct.pack('<Q', len(all_ids)))
            for id_ in all_ids:
                f.write(struct.pack('<H', id_))  # uint16

        print(f'  {split_name}: {len(split_examples)} examples, {len(all_ids):,} tokens → {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='code/domain/')
    parser.add_argument('--out_dir', default='backbone/data/')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--split', type=float, default=0.9)
    args = parser.parse_args()

    prepare(args.data_dir, args.out_dir, args.levels, args.split)
