#!/usr/bin/env python3
"""
Minicalculus corpus generator — Task 1.2 for CRS-1 (Phase 4 extended)
Generates 10k expressions per level (L1-L4), with ground truth via sympy.
Format: {"input": "...", "output": "...", "level": N, "correct": true/false}

Phase 4 additions:
  --exclude_zero_results   corpus-no-zero-result/: filter examples where output == '0'
  --exclude_zero_token     corpus-no-zero-token/: additionally filter any example
                           where '0' appears as a standalone digit in input or output
                           (used with tokenizer-no-zero.py which removes '0' from vocab)
"""
import json, random, sys, argparse
from pathlib import Path
import sympy as sp
from lark import Lark

GRAMMAR = open(Path(__file__).parent / 'grammar.lark').read()
PARSER = Lark(GRAMMAR, start='program', parser='earley', ambiguity='resolve')

VARS = ['x', 'y', 'z', 'n', 'm', 'a', 'b']
SYMS = {v: sp.Symbol(v) for v in VARS}

# ── Generators ───────────────────────────────────────────────────────────────

def rand_int(lo=1, hi=20):
    """Default lo=1 avoids generating 0 naturally (for zero-exclusion variants)."""
    return random.randint(lo, hi)

def rand_num(avoid_zero=False):
    lo = 1 if avoid_zero else 0
    return str(random.randint(lo, 20))

def rand_var():
    return random.choice(VARS[:4])

def rand_l1_expr(depth=0, avoid_zero=False):
    if depth > 3 or random.random() < 0.3:
        return rand_num(avoid_zero=avoid_zero)
    op = random.choice(['+', '-', '*'])
    a, b = rand_l1_expr(depth+1, avoid_zero), rand_l1_expr(depth+1, avoid_zero)
    if op == '*' and depth > 1:
        op = '+'
    return f"({a} {op} {b})"

def rand_l2_expr(depth=0, avoid_zero=False):
    if depth > 3:
        return random.choice([rand_num(avoid_zero=avoid_zero), rand_var()])
    r = random.random()
    if r < 0.25:
        return rand_var()
    if r < 0.35:
        v, args = rand_var(), ', '.join(rand_var() for _ in range(random.randint(1, 2)))
        return f"{v}({args})"
    op = random.choice(['+', '-', '*'])
    a, b = rand_l2_expr(depth+1, avoid_zero), rand_l2_expr(depth+1, avoid_zero)
    return f"({a} {op} {b})"

def rand_l3_stmt(avoid_zero=False):
    kind = random.choice(['assign', 'solve', 'simplify'])
    if kind == 'assign':
        v = rand_var()
        val = random.randint(1 if avoid_zero else 0, 10)
        return f"{v} = {val}", str(val)
    elif kind == 'solve':
        v = rand_var()
        a = random.randint(1, 10)
        # Ensure solution is not zero: b != 0
        b = random.choice([-20, -15, -10, -5, 5, 10, 15, 20])
        lhs = f"{a} * {v} + ({b})"
        sol = sp.Rational(-b, a)
        return f"solve({lhs} == 0, {v})", str(sol)
    else:
        v = rand_var()
        c = random.randint(2, 5)
        return f"simplify({v} + {v})", f"{c}*{v}"

def rand_l4_stmt(avoid_zero=False):
    kind = random.choice(['define', 'forall', 'exists'])
    if kind == 'define':
        fname = random.choice(['f', 'g', 'h'])
        v = rand_var()
        c = random.randint(1, 5)
        return f"define {fname}({v}) = {v} + {c}", f"{v} + {c}"
    elif kind == 'forall':
        v = 'n'
        lo, hi = 1 if avoid_zero else 0, random.randint(3, 10)
        return f"forall {v} in {{{lo} .. {hi}}} : {v} + 0 == {v}", "true"
    else:
        n = random.randint(2, 9)
        sq = n * n
        return f"exists x in {{1 .. {sq+1}}} : x * x == {sq}", "true"

# ── Ground truth ─────────────────────────────────────────────────────────────

def eval_l1(expr_str):
    try:
        val = sp.sympify(expr_str.replace('^', '**'))
        if val.is_number:
            return str(val)
    except:
        pass
    return None

# ── Zero-token check ─────────────────────────────────────────────────────────

def contains_standalone_zero(text):
    """
    Returns True if the string contains '0' as a standalone token
    (i.e., not as part of a multi-digit number like '10', '20').
    Used for corpus-no-zero-token variant.
    """
    import re
    # Match '0' that is not adjacent to another digit
    return bool(re.search(r'(?<!\d)0(?!\d)', text))

# ── Validation ───────────────────────────────────────────────────────────────

def validate(expr_str):
    try:
        PARSER.parse(expr_str)
        return True
    except:
        return False

# ── Corruption (10% incorrect labels) ────────────────────────────────────────

def corrupt_output(correct_out, level, avoid_zero=False):
    try:
        n = int(correct_out)
        delta = random.choice([-2, -1, 1, 2])
        result = n + delta
        if avoid_zero and result == 0:
            result = 1
        return str(result)
    except:
        return correct_out + "_wrong"

# ── Main generation ───────────────────────────────────────────────────────────

def should_exclude(inp, out, exclude_zero_results, exclude_zero_token):
    """
    Phase 4 filtering logic.

    corpus-no-zero-result (exclude_zero_results=True):
        Exclude examples where output token is exactly '0'.
        Preserves '0' elsewhere (digits in inputs, multi-digit numbers).

    corpus-no-zero-token (exclude_zero_token=True, implies exclude_zero_results):
        Exclude examples where '0' appears as a standalone digit ANYWHERE
        in input or output. Model physically cannot produce '0' token.
    """
    if exclude_zero_token:
        # Strict: no standalone '0' anywhere
        if contains_standalone_zero(inp) or contains_standalone_zero(out):
            return True
    elif exclude_zero_results:
        # Lenient: only exclude if output is exactly '0'
        if out.strip() == '0':
            return True
    return False

def generate_level(level, n=1000, exclude_zero_results=False, exclude_zero_token=False):
    avoid_zero = exclude_zero_token  # hint generators to avoid zero
    examples = []
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        try:
            if level == 1:
                inp = rand_l1_expr(avoid_zero=avoid_zero)
                out = eval_l1(inp)
                if out is None:
                    continue
            elif level == 2:
                inp = rand_l2_expr(avoid_zero=avoid_zero)
                out = inp
            elif level == 3:
                inp, out = rand_l3_stmt(avoid_zero=avoid_zero)
            else:
                inp, out = rand_l4_stmt(avoid_zero=avoid_zero)

            if should_exclude(inp, out, exclude_zero_results, exclude_zero_token):
                continue

            correct = random.random() > 0.10
            if not correct:
                out = corrupt_output(out, level, avoid_zero=avoid_zero)
                # Re-check exclusion after corruption
                if should_exclude(inp, out, exclude_zero_results, exclude_zero_token):
                    continue

            if not validate(inp):
                continue

            examples.append({
                "input": inp,
                "output": out,
                "level": level,
                "correct": correct
            })
        except Exception:
            continue

    if len(examples) < n:
        print(f"  Warning: L{level} only generated {len(examples)}/{n} examples", file=sys.stderr)
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minicalculus corpus generator')
    parser.add_argument('--n', type=int, default=10000, help='Examples per level')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: same as script)')
    parser.add_argument('--exclude_zero_results', action='store_true',
                        help='corpus-no-zero-result: exclude examples where output == 0')
    parser.add_argument('--exclude_zero_token', action='store_true',
                        help='corpus-no-zero-token: exclude any example with standalone 0 token')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4])
    args = parser.parse_args()

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.exclude_zero_token:
        out_dir = Path(__file__).parent.parent / 'corpus-no-zero-token'
    elif args.exclude_zero_results:
        out_dir = Path(__file__).parent.parent / 'corpus-no-zero-result'
    else:
        out_dir = Path(__file__).parent

    out_dir.mkdir(parents=True, exist_ok=True)

    variant = 'no-zero-token' if args.exclude_zero_token else \
              'no-zero-result' if args.exclude_zero_results else 'standard'
    print(f'Generating corpus variant: {variant}')
    print(f'Output: {out_dir}')

    total = 0
    zero_counts = {}
    for level in args.levels:
        examples = generate_level(
            level, args.n,
            exclude_zero_results=args.exclude_zero_results,
            exclude_zero_token=args.exclude_zero_token,
        )
        outfile = out_dir / f"corpus-L{level}.jsonl"
        with open(outfile, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        total += len(examples)
        zero_out = sum(1 for ex in examples if ex['output'].strip() == '0')
        zero_counts[level] = zero_out
        print(f"  L{level}: {len(examples)} examples → {outfile} (zero-result outputs: {zero_out})")

    print(f"\nTotal: {total} examples | zero-result outputs: {sum(zero_counts.values())}")
    if args.exclude_zero_results or args.exclude_zero_token:
        assert sum(zero_counts.values()) == 0, "Exclusion filter failed — zero results found!"
        print("✓ Zero-result exclusion verified")
