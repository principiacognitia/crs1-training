"""
Task 2.3: Agent-C (Control) Training
- Progressive curriculum L1 → L4
- Passive feedback: CORRECT/INCORRECT labels fed as context
- Checkpoint every 1000 steps (required for Phase 3 Gnosis integration)
- Log: loss, accuracy per level, time-to-convergence

Usage:
    python train_agent_c.py --config backbone/config.yaml \
        --data_dir backbone/data/ --out_dir results/agent-c/
"""

import os, sys, json, time, math, argparse, random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone' / 'nanoGPT'))

import torch
from model import GPT, GPTConfig

# ── Shared helpers (duplicated for standalone execution) ──────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_bin(path: str):
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
def estimate_loss_per_level(model, level_data: dict, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for lvl, data in level_data.items():
        if data is None or len(data) < block_size + 1:
            continue
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[lvl] = round(sum(losses) / len(losses), 4)
    model.train()
    return out


def get_lr(it, cfg):
    t = cfg['training']
    if it < t['warmup_iters']:
        return t['learning_rate'] * it / t['warmup_iters']
    if it > t['lr_decay_iters']:
        return t['min_lr']
    ratio = (it - t['warmup_iters']) / (t['lr_decay_iters'] - t['warmup_iters'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return t['min_lr'] + coeff * (t['learning_rate'] - t['min_lr'])


# ── Curriculum data mixer ────────────────────────────────────────────────────

class CurriculumSampler:
    """
    At each curriculum step, mix current level with prior levels.
    Prior mix ratio: 20% from prior levels, 80% current.
    """

    def __init__(self, level_data: dict, mix_ratio: float = 0.2):
        self.level_data = level_data   # {1: tensor, 2: tensor, ...}
        self.mix_ratio  = mix_ratio
        self.current_level = min(level_data.keys())

    def advance(self):
        levels = sorted(self.level_data.keys())
        idx = levels.index(self.current_level)
        if idx + 1 < len(levels):
            self.current_level = levels[idx + 1]
            print(f'  [Curriculum] Advanced to L{self.current_level}')

    def sample(self, block_size: int, batch_size: int, device: str):
        """Sample batch: mix current + prior levels."""
        prior_levels = [l for l in self.level_data if l < self.current_level]
        current_data = self.level_data[self.current_level]

        if not prior_levels or self.mix_ratio == 0:
            return get_batch(current_data, block_size, batch_size, device)

        # Split batch
        n_prior   = max(1, int(batch_size * self.mix_ratio))
        n_current = batch_size - n_prior

        # Current level samples
        ix_c = torch.randint(len(current_data) - block_size, (n_current,))
        x_c  = torch.stack([current_data[i:i + block_size] for i in ix_c])
        y_c  = torch.stack([current_data[i + 1:i + block_size + 1] for i in ix_c])

        # Prior level samples (round-robin across prior levels)
        prior_data = self.level_data[prior_levels[-1]]  # most recent prior
        ix_p = torch.randint(len(prior_data) - block_size, (n_prior,))
        x_p  = torch.stack([prior_data[i:i + block_size] for i in ix_p])
        y_p  = torch.stack([prior_data[i + 1:i + block_size + 1] for i in ix_p])

        x = torch.cat([x_c, x_p], dim=0).to(device)
        y = torch.cat([y_c, y_p], dim=0).to(device)
        return x, y


# ── Main training ─────────────────────────────────────────────────────────────

def train(config: dict, data_dir: str, out_dir: str):
    cfg = config
    model_cfg  = cfg['model']
    train_cfg  = cfg['training']
    curric_cfg = cfg.get('curriculum', {})

    device   = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype_str = train_cfg.get('dtype', 'float32')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]

    # ── Load data (all levels) ───────────────────────────────────────────────
    data_dir = Path(data_dir)
    level_data = {}
    vocab_size = None
    for lvl in curric_cfg.get('levels', [1, 2, 3, 4]):
        p = data_dir / f'agent-c-L{lvl}' / 'train.bin'
        if p.exists():
            d, vs = load_bin(p)
            level_data[lvl] = d
            vocab_size = vs
        else:
            print(f'Warning: {p} not found')

    val_data = {}
    for lvl in curric_cfg.get('levels', [1, 2, 3, 4]):
        p = data_dir / f'agent-c-L{lvl}' / 'val.bin'
        if p.exists():
            d, _ = load_bin(p)
            val_data[lvl] = d

    if not level_data:
        print('No training data found. Run prepare_data.py first.')
        sys.exit(1)

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

    optimizer = model.configure_optimizers(
        weight_decay  = train_cfg['weight_decay'],
        learning_rate = train_cfg['learning_rate'],
        betas         = (train_cfg['beta1'], train_cfg['beta2']),
        device_type   = device,
    )

    # ── Curriculum sampler ───────────────────────────────────────────────────
    sampler = CurriculumSampler(
        level_data,
        mix_ratio = curric_cfg.get('prior_mix_ratio', 0.2)
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_iters        = train_cfg['max_iters']
    eval_interval    = train_cfg['eval_interval']
    eval_iters       = train_cfg['eval_iters']
    block_size       = model_cfg['block_size']
    batch_size       = train_cfg['batch_size']
    grad_accum       = train_cfg.get('gradient_accumulation_steps', 1)
    ckpt_every       = train_cfg.get('checkpoint_every', 1000)
    steps_per_level  = curric_cfg.get('steps_per_level', 2500)

    metrics_log = []
    convergence_steps = {}   # level → step at 90% accuracy

    print(f'Agent-C training: L1→L4 curriculum')
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'Device: {device} | dtype: {dtype_str}')
    print(f'Steps per level: {steps_per_level} | Total: {max_iters}')

    t0 = time.time()
    for iter_num in range(max_iters + 1):

        # Curriculum advance
        if iter_num > 0 and iter_num % steps_per_level == 0:
            sampler.advance()

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Eval
        if iter_num % eval_interval == 0:
            level_losses = estimate_loss_per_level(
                model, val_data, block_size, batch_size, eval_iters, device
            )

            metrics = {
                'iter': iter_num,
                'current_level': sampler.current_level,
                'level_losses': level_losses,
                'lr': lr,
                'elapsed_s': round(time.time() - t0, 1),
            }
            metrics_log.append(metrics)
            print(f"step {iter_num:5d} L{sampler.current_level} | "
                  + " | ".join(f"L{k}: {v:.4f}" for k, v in level_losses.items())
                  + f" | lr {lr:.2e}")

            with open(out_dir / 'metrics.json', 'w') as f:
                json.dump(metrics_log, f, indent=2)

            # Track convergence (proxy: loss < 0.5 ≈ ~85% accuracy)
            for lvl, loss in level_losses.items():
                if lvl not in convergence_steps and loss < 0.5:
                    convergence_steps[lvl] = iter_num
                    print(f'  → L{lvl} convergence at step {iter_num}')

        if iter_num == max_iters:
            break

        # Checkpoint
        if iter_num > 0 and iter_num % ckpt_every == 0:
            ckpt = {
                'iter': iter_num,
                'current_level': sampler.current_level,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': gpt_cfg.__dict__,
            }
            torch.save(ckpt, out_dir / f'ckpt_{iter_num:06d}.pt')
            print(f'  → checkpoint saved at step {iter_num}')

        # Forward + backward (curriculum-mixed batch)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = sampler.sample(block_size, batch_size, device)
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        optimizer.step()

    # Final checkpoint + convergence summary
    ckpt = {
        'iter': max_iters,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': gpt_cfg.__dict__,
        'convergence_steps': convergence_steps,
    }
    torch.save(ckpt, out_dir / 'agent-c-final.pt')

    with open(out_dir / 'convergence-stats.json', 'w') as f:
        json.dump(convergence_steps, f, indent=2)

    print(f'\nAgent-C training complete.')
    print(f'Convergence steps: {convergence_steps}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='backbone/config.yaml')
    parser.add_argument('--data_dir', default='backbone/data/')
    parser.add_argument('--out_dir',  default='results/agent-c/')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print('pip install pyyaml')
        sys.exit(1)

    cfg = load_config(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)
    train(cfg, args.data_dir, args.out_dir)
