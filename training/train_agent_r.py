"""
Task 2.2: Agent-R (Restricted) Training
- Train on L1 corpus only
- Freeze weights after L1 convergence
- Evaluate OOD performance on L2-L4
- Log: loss curves, OOD accuracy per level

Usage:
    python train_agent_r.py --config backbone/config.yaml \
        --data_dir backbone/data/ --out_dir results/agent-r/
"""

import os, sys, json, time, math, argparse
from pathlib import Path

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone' / 'nanoGPT'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── nanoGPT imports ──────────────────────────────────────────────────────────
from model import GPT, GPTConfig


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_bin(path: str) -> torch.Tensor:
    """Load uint16 binary token file produced by prepare_data.py"""
    import struct
    with open(path, 'rb') as f:
        vocab_size = struct.unpack('<I', f.read(4))[0]
        n_tokens   = struct.unpack('<Q', f.read(8))[0]
        ids = torch.frombuffer(f.read(n_tokens * 2), dtype=torch.uint16).long()
    return ids, vocab_size


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data_splits, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for name, data in data_splits.items():
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def eval_ood_accuracy(model, data: torch.Tensor, block_size: int, batch_size: int,
                      eos_id: int, device: str, n_batches: int = 50) -> float:
    """Evaluate next-token prediction accuracy as OOD proxy."""
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        logits, _ = model(x, targets=None)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total if total > 0 else 0.0


def train(config: dict, data_dir: str, out_dir: str):
    cfg = config
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    device = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype_str = train_cfg.get('dtype', 'float32')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]

    # ── Load data (L1 only) ──────────────────────────────────────────────────
    data_dir = Path(data_dir)
    # Expect: agent-r/train.bin, agent-r/val.bin (L1 only)
    train_data, vocab_size = load_bin(data_dir / 'agent-r' / 'train.bin')
    val_data, _            = load_bin(data_dir / 'agent-r' / 'val.bin')

    # OOD eval sets for L2-L4
    ood_data = {}
    for lvl in [2, 3, 4]:
        p = data_dir / f'agent-r-ood-L{lvl}' / 'val.bin'
        if p.exists():
            ood_data[f'L{lvl}'], _ = load_bin(p)

    # ── Model ────────────────────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        n_layer    = model_cfg['n_layer'],
        n_head     = model_cfg['n_head'],
        n_embd     = model_cfg['n_embd'],
        block_size = model_cfg['block_size'],
        bias       = model_cfg.get('bias', False),
        vocab_size = vocab_size,
        dropout    = model_cfg.get('dropout', 0.1),
    )
    model = GPT(gpt_cfg).to(device)
    if train_cfg.get('compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = model.configure_optimizers(
        weight_decay  = train_cfg['weight_decay'],
        learning_rate = train_cfg['learning_rate'],
        betas         = (train_cfg['beta1'], train_cfg['beta2']),
        device_type   = device,
    )

    # ── LR schedule ──────────────────────────────────────────────────────────
    def get_lr(it):
        if it < train_cfg['warmup_iters']:
            return train_cfg['learning_rate'] * it / train_cfg['warmup_iters']
        if it > train_cfg['lr_decay_iters']:
            return train_cfg['min_lr']
        ratio = (it - train_cfg['warmup_iters']) / (train_cfg['lr_decay_iters'] - train_cfg['warmup_iters'])
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return train_cfg['min_lr'] + coeff * (train_cfg['learning_rate'] - train_cfg['min_lr'])

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []

    print(f'Agent-R training: L1 only → OOD eval on L2-L4')
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'Device: {device} | dtype: {dtype_str}')
    print(f'Train tokens: {len(train_data):,}')

    block_size  = model_cfg['block_size']
    batch_size  = train_cfg['batch_size']
    grad_accum  = train_cfg.get('gradient_accumulation_steps', 1)
    max_iters   = train_cfg['max_iters']
    eval_iters  = train_cfg['eval_iters']
    eval_interval = train_cfg['eval_interval']
    ckpt_every  = train_cfg.get('checkpoint_every', 1000)

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.time()
    for iter_num in range(max_iters + 1):

        # Update LR
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Eval
        if iter_num % eval_interval == 0:
            losses = estimate_loss(
                model,
                {'train': train_data, 'val': val_data},
                block_size, batch_size, eval_iters, device
            )
            ood_acc = {k: eval_ood_accuracy(model, v, block_size, batch_size, -1, device)
                       for k, v in ood_data.items()}

            metrics = {
                'iter': iter_num,
                'train_loss': round(losses['train'], 4),
                'val_loss': round(losses['val'], 4),
                'ood_accuracy': {k: round(v, 4) for k, v in ood_acc.items()},
                'lr': lr,
                'elapsed_s': round(time.time() - t0, 1),
            }
            metrics_log.append(metrics)
            print(f"step {iter_num:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| OOD {ood_acc} | lr {lr:.2e}")

            with open(out_dir / 'metrics.json', 'w') as f:
                json.dump(metrics_log, f, indent=2)

        if iter_num == max_iters:
            break

        # Checkpoint
        if iter_num > 0 and iter_num % ckpt_every == 0:
            ckpt = {
                'iter': iter_num,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': gpt_cfg.__dict__,
            }
            torch.save(ckpt, out_dir / f'ckpt_{iter_num:06d}.pt')
            print(f'  → checkpoint saved at step {iter_num}')

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum):
            x, y = get_batch(train_data, block_size, batch_size, device)
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        optimizer.step()

    # Final save
    ckpt = {
        'iter': max_iters,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': gpt_cfg.__dict__,
        'frozen': True,   # Agent-R weights are frozen after L1 training
    }
    torch.save(ckpt, out_dir / 'agent-r-final.pt')
    print(f'\nAgent-R training complete. Final checkpoint: {out_dir}/agent-r-final.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='backbone/config.yaml')
    parser.add_argument('--data_dir', default='backbone/data/')
    parser.add_argument('--out_dir',  default='results/agent-r/')
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print('pip install pyyaml')
        sys.exit(1)

    cfg = load_config(args.config)
    train(cfg, args.data_dir, args.out_dir)
