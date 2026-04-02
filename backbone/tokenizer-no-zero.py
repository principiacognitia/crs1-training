"""
MinicalcTokenizer — No-Zero variant for CRS-1 Phase 4 zero-result exclusion test.

Removes '0' from DIGITS and NUMBER_TOKENS so the model physically cannot
output the zero token. Used with corpus-no-zero-token/ corpus variant.

Rationale: If the model is trained entirely without the '0' token and then
probed on expressions whose correct answer is zero (e.g. '3 - 3', 'x + 5 = 5'),
low Gnosis p_avg on these probes indicates the gate tracks genuine OOD uncertainty
rather than just training-distribution confidence.

Usage:
    from tokenizer_no_zero import MinicalcTokenizerNoZero as MinicalcTokenizer
    tok = MinicalcTokenizer()

Changes from tokenizer.py:
- '0' removed from DIGITS (single-digit zero absent)
- '0' removed from NUMBER_TOKENS (standalone zero string absent)
- Multi-digit tokens containing zero ('10', '20', ..., '90') are KEPT —
  they are distinct tokens and do not represent standalone zero
- Vocab size: 176 (standard) → 175 (no-zero variant; -1 for removed '0')
"""

import json
from pathlib import Path

# ── Vocabulary Definition ─────────────────────────────────────────────────────

SPECIAL = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>']

# '0' removed — model cannot output standalone zero
DIGITS = [str(i) for i in range(1, 10)]  # 1-9 only

LOWERCASE = [chr(c) for c in range(ord('a'), ord('z') + 1)]

OPERATORS = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '^']

PUNCTUATION = ['(', ')', ',', '.', ';', ':', ' ', '\n']

KEYWORDS = [
    'solve', 'simplify', 'diff', 'integrate',
    'for', 'all', 'exists', 'in', 'not', 'and', 'or',
    'if', 'then', 'else', 'def', 'return',
    'True', 'False',
]

# Multi-char number tokens — '0' as standalone excluded, but '10', '20', etc. kept
# (they are distinct multi-char tokens, not the zero digit)
NUMBER_TOKENS = [str(i) for i in range(1, 100)]  # 1-99; 0 excluded

# Greek/math symbols for L4
MATH_SYMBOLS = ['∀', '∃', '∈', '→', '↔', '∧', '∨', '¬']

ALL_TOKENS = (
    SPECIAL
    + DIGITS
    + LOWERCASE
    + OPERATORS
    + PUNCTUATION
    + KEYWORDS
    + NUMBER_TOKENS
    + MATH_SYMBOLS
)

seen = set()
VOCAB = []
for t in ALL_TOKENS:
    if t not in seen:
        VOCAB.append(t)
        seen.add(t)

assert '0' not in VOCAB, "No-zero tokenizer still contains '0'"

TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB)}

VOCAB_SIZE = len(VOCAB)

PAD_ID  = TOKEN_TO_ID['<PAD>']
BOS_ID  = TOKEN_TO_ID['<BOS>']
EOS_ID  = TOKEN_TO_ID['<EOS>']
UNK_ID  = TOKEN_TO_ID['<UNK>']


class MinicalcTokenizerNoZero:
    """
    No-zero variant of MinicalcTokenizer.
    '0' maps to <UNK> if encountered — which should never happen with
    the no-zero-token corpus variant.
    """

    def __init__(self):
        self.vocab = VOCAB
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        self.vocab_size = VOCAB_SIZE
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
                ids.append(UNK_ID)  # '0' will land here
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
                'variant': 'no-zero',
                'special_ids': {
                    'pad': PAD_ID, 'bos': BOS_ID,
                    'eos': EOS_ID, 'unk': UNK_ID,
                },
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MinicalcTokenizerNoZero':
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


# Alias for drop-in replacement
MinicalcTokenizer = MinicalcTokenizerNoZero


if __name__ == '__main__':
    tok = MinicalcTokenizerNoZero()
    print(f'Vocab size: {tok.vocab_size} (standard: 176, no-zero: {tok.vocab_size})')
    print(f'"0" in vocab: {"0" in tok.token_to_id}')

    tests = [
        ('10 + 5', True),      # '10' is fine (multi-digit token), result is 15
        ('3 - 3', False),      # result is 0 — should never appear in training
        ('x + 5', True),
    ]
    for text, expect_clean in tests:
        enc = tok.encode(text)
        dec = tok.decode(enc)
        has_unk = UNK_ID in enc
        print(f'  {"✓" if (not has_unk) == expect_clean else "✗"} {text!r} → {dec!r} (UNK: {has_unk})')

    print(f'\nProbe set (should all produce UNK/failure on no-zero model):')
    probes = ['3 - 3', 'x + 5 = 5', 'simplify(a - a)']
    for p in probes:
        enc = tok.encode(p)
        print(f'  {p!r}: {len(enc)} tokens, contains UNK: {UNK_ID in enc}')
