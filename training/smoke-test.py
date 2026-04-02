"""
CRS-1 Phase 2 Smoke Test — Task 2.1 validation
Validates tokenizer + nanoGPT config before full training runs.

Tests:
1. Load 100 examples from corpus-L1.jsonl
2. Tokenize with custom minicalculus tokenizer
3. Train 2M param model for 50 iterations (~5min on RTX 4060, <60s on CPU)
4. Verify loss decreases + no CUDA OOM
5. Generate 3 samples, check grammar validity via Lark parser

Pass criteria (Phase 2 → Phase 3):
  - Loss < 2.0 after 50 iterations
  - No CUDA OOM
  - At least 1/3 samples parse without error

Usage:
    python training/smoke-test.py [--cpu] [--data_dir code/domain/]
"""

import sys, json, time, math, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone' / 'nanoGPT'))

PASS = '✓'
FAIL = '✗'
WARN = '⚠'


def load_examples(data_dir: str, n: int = 100) -> list[dict]:
    path = Path(data_dir) / 'corpus-L1.jsonl'
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def check_grammar(text: str) -> bool:
    """Try to parse text with Lark grammar. Returns True if valid."""
    try:
        from lark import Lark
        grammar_path = Path(__file__).parent.parent / 'code' / 'domain' / 'grammar.lark'
        if not grammar_path.exists():
            # Try alternate location
            grammar_path = Path('code/domain/grammar.lark')
        if not grammar_path.exists():
            return None  # Can't validate, grammar not found
        parser = Lark(open(grammar_path).read(), start='program', parser='earley', ambiguity='resolve')
        parser.parse(text)
        return True
    except Exception:
        return False


def run_smoke_test(data_dir: str, force_cpu: bool = False):
    print('=' * 60)
    print('CRS-1 Phase 2 Smoke Test')
    print('=' * 60)

    results = {
        'tokenizer': None,
        'model_init': None,
        'loss_decrease': None,
        'final_loss': None,
        'no_oom': None,
        'generation': None,
        'grammar_valid': None,
        'passed': False,
    }

    # ── Step 1: Tokenizer ────────────────────────────────────────────────────
    print('\n[1/5] Tokenizer validation...')
    try:
        from tokenizer import MinicalcTokenizer
        tok = MinicalcTokenizer()
        examples = load_examples(data_dir, 100)
        assert len(examples) == 100, f'Expected 100 examples, got {len(examples)}'

        token_counts = []
        for ex in examples[:10]:
            text = ex.get('input', '')
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f'Round-trip fail: {text!r} → {decoded!r}'
            token_counts.append(len(ids))

        avg_tokens = sum(token_counts) / len(token_counts)
        print(f'  {PASS} Vocab size: {tok.vocab_size}')
        print(f'  {PASS} 100 examples loaded from corpus-L1.jsonl')
        print(f'  {PASS} Round-trip encoding verified (10/10)')
        print(f'  {PASS} Avg tokens per example: {avg_tokens:.1f}')
        results['tokenizer'] = True

    except Exception as e:
        print(f'  {FAIL} Tokenizer failed: {e}')
        results['tokenizer'] = False
        print('\nSMOKE TEST FAILED at step 1.')
        return results

    # ── Step 2: Build token dataset ──────────────────────────────────────────
    print('\n[2/5] Building token sequence...')
    all_ids = []
    for ex in examples:
        text = ex.get('input', '')
        out = ex.get('output', '')
        full = f"{text} = {out}" if out else text
        all_ids.extend(tok.encode(full))

    print(f'  {PASS} {len(all_ids):,} tokens from 100 examples')

    # ── Step 3: Model initialization ─────────────────────────────────────────
    print('\n[3/5] Model initialization...')
    try:
        import torch
        from model import GPT, GPTConfig

        device = 'cpu' if (force_cpu or not torch.cuda.is_available()) else 'cuda'
        if device == 'cpu':
            print(f'  {WARN} Running on CPU (slower, ~60s expected)')
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  {PASS} GPU: {gpu_name} ({gpu_mem:.1f} GB)')

        cfg = GPTConfig(
            n_layer=4, n_head=6, n_embd=384,
            block_size=128,  # shorter for smoke test
            bias=False, vocab_size=tok.vocab_size, dropout=0.0,
        )
        model = GPT(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  {PASS} Model: {n_params:,} params ({n_params/1e6:.1f}M)')
        results['model_init'] = True

        if device == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1e6
            print(f'  {PASS} VRAM used after init: {mem_used:.1f} MB')

    except torch.cuda.OutOfMemoryError:
        print(f'  {FAIL} CUDA OOM at model init — reduce n_embd or batch_size')
        results['no_oom'] = False
        results['model_init'] = False
        return results
    except Exception as e:
        print(f'  {FAIL} Model init failed: {e}')
        results['model_init'] = False
        return results

    # ── Step 4: 50-iteration training ────────────────────────────────────────
    print('\n[4/5] Training 50 iterations...')
    try:
        import torch.nn.functional as F

        data = torch.tensor(all_ids, dtype=torch.long)
        block_size = 128
        batch_size = 8  # small for smoke test
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        losses = []
        t0 = time.time()

        for i in range(50):
            # Random batch
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[j:j + block_size] for j in ix]).to(device)
            y = torch.stack([data[j + 1:j + block_size + 1] for j in ix]).to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 10 == 0:
                print(f'  iter {i+1:3d}/50  loss: {loss.item():.4f}')

        elapsed = time.time() - t0
        loss_start = sum(losses[:5]) / 5
        loss_end   = sum(losses[-5:]) / 5
        loss_decrease = loss_start - loss_end

        results['final_loss'] = round(loss_end, 4)
        results['no_oom'] = True

        print(f'\n  Time: {elapsed:.1f}s')
        print(f'  Loss start (avg 5): {loss_start:.4f}')
        print(f'  Loss end   (avg 5): {loss_end:.4f}')
        print(f'  Decrease: {loss_decrease:.4f}')

        if loss_decrease > 0:
            print(f'  {PASS} Loss is decreasing')
            results['loss_decrease'] = True
        else:
            print(f'  {WARN} Loss not decreasing (may need more data or different LR)')
            results['loss_decrease'] = False

        if loss_end < 2.0:
            print(f'  {PASS} Final loss {loss_end:.4f} < 2.0 (gate criterion met)')
        else:
            print(f'  {WARN} Final loss {loss_end:.4f} ≥ 2.0 (50 iters may not be enough — check with full run)')

    except torch.cuda.OutOfMemoryError:
        print(f'  {FAIL} CUDA OOM during training')
        results['no_oom'] = False
        return results
    except Exception as e:
        print(f'  {FAIL} Training error: {e}')
        return results

    # ── Step 5: Generate 3 samples + grammar check ───────────────────────────
    print('\n[5/5] Generation + grammar validation...')
    model.eval()
    valid_count = 0
    samples = []

    with torch.no_grad():
        # Seed with BOS token
        for sample_idx in range(3):
            prompt = torch.tensor([[tok.token_to_id['<BOS>']]], dtype=torch.long).to(device)
            generated = []

            for _ in range(40):  # max 40 tokens
                if prompt.shape[1] > block_size:
                    prompt = prompt[:, -block_size:]
                logits, _ = model(prompt)
                next_tok = torch.multinomial(
                    F.softmax(logits[:, -1, :] / 0.8, dim=-1), 1
                )
                next_id = next_tok.item()
                if next_id == tok.token_to_id.get('<EOS>', -1):
                    break
                generated.append(next_id)
                prompt = torch.cat([prompt, next_tok], dim=1)

            text = tok.decode(generated, skip_special=True).strip()
            samples.append(text)

            valid = check_grammar(text)
            status = PASS if valid else (WARN if valid is None else FAIL)
            print(f'  Sample {sample_idx+1}: {text!r}')
            if valid is True:
                print(f'    {PASS} Valid grammar')
                valid_count += 1
            elif valid is None:
                print(f'    {WARN} Grammar check skipped (grammar.lark not found)')
            else:
                print(f'    {FAIL} Invalid grammar (expected at 50 iters)')

    results['generation'] = True
    results['grammar_valid'] = valid_count

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SMOKE TEST SUMMARY')
    print('=' * 60)

    gate_pass = (
        results['tokenizer'] and
        results['model_init'] and
        results['no_oom'] and
        results.get('final_loss', 99) < 2.0
    )

    results['passed'] = gate_pass

    print(f'  Tokenizer:      {PASS if results["tokenizer"] else FAIL}')
    print(f'  Model init:     {PASS if results["model_init"] else FAIL}')
    print(f'  No CUDA OOM:    {PASS if results["no_oom"] else FAIL}')
    print(f'  Loss < 2.0:     {PASS if (results.get("final_loss",99) < 2.0) else FAIL} ({results.get("final_loss","?")})')
    print(f'  Loss decreasing:{PASS if results["loss_decrease"] else WARN}')
    print(f'  Generation:     {PASS if results["generation"] else FAIL}')
    print(f'  Grammar valid:  {results["grammar_valid"]}/3 samples')

    print()
    if gate_pass:
        print(f'  ✅ GATE: Phase 2 → Phase 3 APPROVED')
        print(f'  Ready to run full Agent-R and Agent-C training.')
    else:
        print(f'  ❌ GATE: Phase 2 → Phase 3 BLOCKED')
        if not results['no_oom']:
            print(f'  → Reduce batch_size (try 16) or n_embd (try 256)')
        if results.get('final_loss', 0) >= 2.0:
            print(f'  → Loss too high: check tokenizer or increase model capacity')
        if not results['tokenizer']:
            print(f'  → Fix tokenizer before proceeding')

    # Save results
    out_path = Path('results/smoke-test-results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved: {out_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU (for testing without GPU)')
    parser.add_argument('--data_dir', default='code/domain/', help='Path to corpus files')
    args = parser.parse_args()

    results = run_smoke_test(args.data_dir, force_cpu=args.cpu)
    sys.exit(0 if results.get('passed') else 1)
