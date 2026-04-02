"""
Minicalculus custom tokenizer — Task 2.1 for CRS-1
Character-level vocab built from the minicalculus grammar, NOT BPE.
Covers all tokens needed for L1-L4: operators, keywords, identifiers, digits, punctuation.
"""

import json
from pathlib import Path

# ── Vocabulary Definition ─────────────────────────────────────────────────────
# Tokens ordered by type for readability; IDs assigned sequentially.

SPECIAL = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>']

DIGITS = [str(i) for i in range(10)]

LOWERCASE = [chr(c) for c in range(ord('a'), ord('z') + 1)]

OPERATORS = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '^']

PUNCTUATION = ['(', ')', ',', '.', ';', ':', ' ', '\n']

KEYWORDS = [
    'solve', 'simplify', 'diff', 'integrate',
    'for', 'all', 'exists', 'in', 'not', 'and', 'or',
    'if', 'then', 'else', 'def', 'return',
    'True', 'False',
]

# Multi-char number tokens (0-99) for efficiency
NUMBER_TOKENS = [str(i) for i in range(100)]

# Greek/math symbols for L4
MATH_SYMBOLS = ['∀', '∃', '∈', '→', '↔', '∧', '∨', '¬']

# Build vocab
ALL_TOKENS = (
    SPECIAL
    + DIGITS
    + LOWERCASE
    + OPERATORS
    + PUNCTUATION
    + KEYWORDS
    + NUMBER_TOKENS  # two-char numbers; duplicates resolved below
    + MATH_SYMBOLS
)

# Deduplicate while preserving order
seen = set()
VOCAB = []
for t in ALL_TOKENS:
    if t not in seen:
        VOCAB.append(t)
        seen.add(t)

TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB)}

VOCAB_SIZE = len(VOCAB)

PAD_ID  = TOKEN_TO_ID['<PAD>']
BOS_ID  = TOKEN_TO_ID['<BOS>']
EOS_ID  = TOKEN_TO_ID['<EOS>']
UNK_ID  = TOKEN_TO_ID['<UNK>']


class MinicalcTokenizer:
    """
    Greedy longest-match tokenizer for minicalculus.
    No subword splitting — vocab designed to cover all legal tokens exactly.
    """

    def __init__(self):
        self.vocab = VOCAB
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        self.vocab_size = VOCAB_SIZE
        # Sort by length descending for greedy longest match
        self._sorted_tokens = sorted(
            [t for t in VOCAB if t not in ('<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>')],
            key=len, reverse=True
        )

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [BOS_ID] if add_special else []
        i = 0
        while i < len(text):
            matched = False
            for tok in self._sorted_tokens:
                if text[i:i + len(tok)] == tok:
                    ids.append(TOKEN_TO_ID[tok])
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(UNK_ID)
                i += 1
        if add_special:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        parts = []
        for i in ids:
            tok = ID_TO_TOKEN.get(i, '<UNK>')
            if skip_special and tok in ('<PAD>', '<BOS>', '<EOS>', '<MASK>'):
                continue
            parts.append(tok)
        return ''.join(parts)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'vocab': VOCAB,
                'token_to_id': TOKEN_TO_ID,
                'vocab_size': VOCAB_SIZE,
                'special_ids': {
                    'pad': PAD_ID, 'bos': BOS_ID,
                    'eos': EOS_ID, 'unk': UNK_ID,
                },
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MinicalcTokenizer':
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.vocab = data['vocab']
        tok.token_to_id = {t: int(i) for t, i in data['token_to_id'].items()}
        tok.id_to_token = {int(i): t for t, i in data['token_to_id'].items()}
        tok.vocab_size = data['vocab_size']
        tok._sorted_tokens = sorted(
            [t for t in tok.vocab if t not in ('<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>')],
            key=len, reverse=True
        )
        return tok


# ── VRAM estimate ─────────────────────────────────────────────────────────────

def estimate_vram_mb(
    vocab_size: int = VOCAB_SIZE,
    n_layer: int = 4,
    n_embd: int = 384,
    n_head: int = 6,
    block_size: int = 256,
    batch_size: int = 32,
    dtype_bytes: int = 4,  # float32; use 2 for bf16
) -> dict:
    """Rough VRAM estimate for nanoGPT config."""
    # Embedding table
    embed = vocab_size * n_embd * dtype_bytes
    # Per-layer: attn QKV + proj + MLP (4x + 4x) ≈ 12 * n_embd^2
    per_layer = 12 * n_embd * n_embd * dtype_bytes
    params_bytes = embed + n_layer * per_layer
    params_mb = params_bytes / 1e6

    # Activations: batch * seq_len * n_embd * n_layer * ~2 (fwd + bwd)
    activations_mb = batch_size * block_size * n_embd * n_layer * 2 * dtype_bytes / 1e6

    total_mb = params_mb + activations_mb
    total_params = (embed // dtype_bytes) + n_layer * (12 * n_embd * n_embd)

    return {
        'total_params': total_params,
        'params_mb': round(params_mb, 1),
        'activations_mb': round(activations_mb, 1),
        'total_mb': round(total_mb, 1),
        'fits_8gb': total_mb < 8000,
    }


if __name__ == '__main__':
    tok = MinicalcTokenizer()
    print(f'Vocab size: {tok.vocab_size}')

    # Test round-trip
    tests = [
        '(3 + 5)',
        'x = 10',
        'solve(x + 5 == 10)',
        'simplify((x * 2) + 0)',
        'for all x in (0, 10)',
    ]
    for t in tests:
        enc = tok.encode(t)
        dec = tok.decode(enc)
        ok = dec == t
        print(f'  {"✓" if ok else "✗"} [{len(enc)} tokens] {t!r}')

    vram = estimate_vram_mb()
    print(f'\nVRAM estimate (4-layer, d=384, batch=32):')
    print(f'  Params: {vram["total_params"]:,} ({vram["params_mb"]} MB)')
    print(f'  Activations: {vram["activations_mb"]} MB')
    print(f'  Total: {vram["total_mb"]} MB  →  fits 8GB: {vram["fits_8gb"]}')
