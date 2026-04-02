"""
Agent-Static-Mix: Fixed 50/50 curriculum baseline.
- Hard curriculum transitions (same schedule as Agent-C)
- Fixed 50/50 prior/current mix throughout — no adaptive weighting
- No Gnosis gate — pure fixed-mix control condition
- Purpose: isolate contribution of adaptive gating vs fixed mixing strategy

Hypothesis: if Agent-N-C >> Agent-Static-Mix, the gate adds value beyond
simply mixing prior levels. If Agent-Static-Mix ≈ Agent-N-C, mixing ratio
alone explains the improvement.

Usage:
    python training/train_agent_static.py --config backbone/config.yaml \
        --data_dir backbone/data/ --out_dir results/agent-static/
"""

import os, sys, json, time, math, argparse, random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone' / 'nanoGPT'))

import torch
from model import GPT, GPTConfig


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


class StaticMixSampler:
    """
    Fixed 50/50 prior/current mix. No adaptive weighting.
    Hard transitions at fixed step intervals (same schedule as Agent-C).
    Prior levels share the 50% prior budget equally.
    """

    STATIC_MIX_RATIO = 0.5  # fixed: 50% prior, 50% current

    def __init__(self, level_data: dict):
        self.level_data    = level_data
        self.current_level = min(level_data.keys())

    def advance(self):
        levels = sorted(self.level_data.keys())
        idx = levels.index(self.current_level)
        if idx + 1 < len(levels):
            self.current_level = levels[idx + 1]
            print(f'  [StaticMix] Advanced to L{self.current_level}')

    def sample(self, block_size: int, batch_size: int, device: str):
        prior_levels = [l for l in self.level_data if l < self.current_level]
        current_data = self.level_data[self.current_level]

        if not prior_levels:
            # At L1: no prior — pure current
            return get_batch(current_data, block_size, batch_size, device)

        # Fixed 50/50 split
        n_prior   = batch_size // 2
        n_current = batch_size - n_prior

        # Current level samples
        ix_c = torch.randint(len(current_data) - block_size, (n_current,))
        x_c  = torch.stack([current_data[i:i + block_size] for i in ix_c])
        y_c  = torch.stack([current_data[i + 1:i + block_size + 1] for i in ix_c])

        # Prior levels: distribute evenly across all prior levels
        n_per_prior = max(1, n_prior // len(prior_levels))
        xs_p, ys_p = [], []
        for lvl in prior_levels:
            pdata = self.level_data[lvl]
            n = n_per_prior if lvl != prior_levels[-1] else n_prior - n_per_prior * (len(prior_levels) - 1)
            if n <= 0:
                continue
            ix_p = torch.randint(len(pdata) - block_size, (n,))
            xs_p.append(torch.stack([pdata[i:i + block_size] for i in ix_p]))
            ys_p.append(torch.stack([pdata[i + 1:i + block_size + 1] for i in ix_p]))

        x_p = torch.cat(xs_p, dim=0) if xs_p else x_c[:0]
        y_p = torch.cat(ys_p, dim=0) if ys_p else y_c[:0]

        x = torch.cat([x_c, x_p], dim=0).to(device)
        y = torch.cat([y_c, y_p], dim=0).to(device)
        return x, y


def train(config: dict, data_dir: str, out_dir: str):
    cfg        = config
    model_cfg  = cfg['model']
    train_cfg  = cfg['training']
    curric_cfg = cfg.get('curriculum', {})

    device    = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype_str = train_cfg.get('dtype', 'float32')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype_str]

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

    sampler = StaticMixSampler(level_data)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_iters       = train_cfg['max_iters']
    eval_interval   = train_cfg['eval_interval']
    eval_iters      = train_cfg['eval_iters']
    block_size      = model_cfg['block_size']
    batch_size      = train_cfg['batch_size']
    grad_accum      = train_cfg.get('gradient_accumulation_steps', 1)
    ckpt_every      = train_cfg.get('checkpoint_every', 1000)
    steps_per_level = curric_cfg.get('steps_per_level', 2500)

    metrics_log = []
    convergence_steps = {}

    print(f'Agent-Static-Mix training: fixed 50/50 prior/current, no gate')
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'Device: {device} | dtype: {dtype_str}')
    print(f'Steps per level: {steps_per_level} | Total: {max_iters}')

    t0 = time.time()
    for iter_num in range(max_iters + 1):

        if iter_num > 0 and iter_num % steps_per_level == 0:
            sampler.advance()

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

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

            for lvl, loss in level_losses.items():
                if lvl not in convergence_steps and loss < 0.5:
                    convergence_steps[lvl] = iter_num

        if iter_num == max_iters:
            break

        if iter_num > 0 and iter_num % ckpt_every == 0:
            ckpt = {
                'iter': iter_num,
                'current_level': sampler.current_level,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': gpt_cfg.__dict__,
            }
            torch.save(ckpt, out_dir / f'ckpt_{iter_num:06d}.pt')

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = sampler.sample(block_size, batch_size, device)
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        optimizer.step()

    ckpt = {
        'iter': max_iters,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': gpt_cfg.__dict__,
        'convergence_steps': convergence_steps,
    }
    torch.save(ckpt, out_dir / 'agent-static-final.pt')

    with open(out_dir / 'convergence-stats.json', 'w') as f:
        json.dump(convergence_steps, f, indent=2)

    print(f'\nAgent-Static-Mix training complete.')
    print(f'Convergence steps: {convergence_steps}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='backbone/config.yaml')
    parser.add_argument('--data_dir', default='backbone/data/')
    parser.add_argument('--out_dir',  default='results/agent-static/')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print('pip install pyyaml')
        sys.exit(1)

    cfg = load_config(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(cfg, args.data_dir, args.out_dir)
