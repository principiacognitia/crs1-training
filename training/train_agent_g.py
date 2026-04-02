"""
Task 3-G: Agent-G (Gradual) Training
- Progressive L1→L4 curriculum with GRADUAL mixing ramp (no Gnosis gate)
- Tests whether smooth level introduction alone prevents catastrophic forgetting
- Condition 2 in three-condition experiment (Agent-C / Agent-G / Agent-N)

Ramp logic: at each level transition, new level starts at 20% of batch,
ramps linearly to 80% over `ramp_steps` steps. Prior levels decay symmetrically.
ramp_steps = 500 (V_G recovery timescale from Agent-C Phase 2 results)

Usage:
    python train_agent_g.py --config backbone/config.yaml \
        --data_dir backbone/data/ --out_dir results/agent-g/
"""

import os, sys, json, time, math, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone' / 'nanoGPT'))

import torch
from model import GPT, GPTConfig


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_bin(path):
    import struct
    with open(path, 'rb') as f:
        vocab_size = struct.unpack('<I', f.read(4))[0]
        n_tokens   = struct.unpack('<Q', f.read(8))[0]
        ids = torch.frombuffer(f.read(n_tokens * 2), dtype=torch.uint16).long()
    return ids, vocab_size


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss_per_level(model, val_data, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for lvl, data in val_data.items():
        if data is None or len(data) < block_size + 1:
            continue
        losses = [model(*get_batch(data, block_size, batch_size, device))[1].item()
                  for _ in range(eval_iters)]
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


# ── Gradual Curriculum Sampler ────────────────────────────────────────────────

class GradualCurriculumSampler:
    """
    At each level transition, ramps new level weight from 20% → 80% over ramp_steps.
    Prior levels share the remaining weight symmetrically.

    This directly tests whether smooth introduction prevents forgetting
    without any gating mechanism (Agent-G condition).
    """

    def __init__(self, level_data: dict, ramp_steps: int = 500):
        self.level_data = level_data
        self.ramp_steps = ramp_steps
        self.levels = sorted(level_data.keys())
        self.current_level_idx = 0
        self.level_start_step = 0
        self.current_step = 0

    @property
    def current_level(self):
        return self.levels[self.current_level_idx]

    def advance(self):
        if self.current_level_idx + 1 < len(self.levels):
            self.current_level_idx += 1
            self.level_start_step = self.current_step
            print(f'  [GradualCurriculum] Advanced to L{self.current_level} at step {self.current_step}')

    def get_weights(self) -> dict:
        """
        Returns per-level sampling weights.
        New level ramps 0.2 → 0.8 over ramp_steps.
        Prior levels share remaining weight equally.
        """
        steps_into_level = self.current_step - self.level_start_step
        new_weight = min(0.8, 0.2 + (steps_into_level / self.ramp_steps) * 0.6)

        active_levels = self.levels[:self.current_level_idx + 1]
        prior_levels = active_levels[:-1]

        weights = {}
        if not prior_levels:
            weights[self.current_level] = 1.0
        else:
            prior_weight = 1.0 - new_weight
            per_prior = prior_weight / len(prior_levels)
            weights[self.current_level] = new_weight
            for lvl in prior_levels:
                weights[lvl] = per_prior

        return weights

    def sample(self, block_size, batch_size, device):
        """Sample a mixed batch according to current ramp weights."""
        weights = self.get_weights()
        samples_x, samples_y = [], []

        for lvl, w in weights.items():
            n = max(1, round(w * batch_size))
            data = self.level_data[lvl]
            ix = torch.randint(len(data) - block_size, (n,))
            samples_x.extend([data[i:i + block_size] for i in ix])
            samples_y.extend([data[i + 1:i + block_size + 1] for i in ix])

        # Trim/pad to exact batch_size
        samples_x = samples_x[:batch_size]
        samples_y = samples_y[:batch_size]

        x = torch.stack(samples_x).to(device)
        y = torch.stack(samples_y).to(device)
        return x, y


# ── Main training ─────────────────────────────────────────────────────────────

def train(config, data_dir, out_dir):
    cfg = config
    model_cfg  = cfg['model']
    train_cfg  = cfg['training']
    curric_cfg = cfg.get('curriculum', {})

    device    = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype_str = train_cfg.get('dtype', 'float32')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype_str]

    data_dir = Path(data_dir)
    level_data, val_data = {}, {}
    vocab_size = None
    for lvl in curric_cfg.get('levels', [1, 2, 3, 4]):
        p = data_dir / f'agent-c-L{lvl}' / 'train.bin'  # reuse Agent-C data
        if p.exists():
            d, vs = load_bin(p); level_data[lvl] = d; vocab_size = vs
        p = data_dir / f'agent-c-L{lvl}' / 'val.bin'
        if p.exists():
            d, _ = load_bin(p); val_data[lvl] = d

    if not level_data:
        print('No training data. Run prepare_data.py first.'); sys.exit(1)

    gpt_cfg = GPTConfig(
        n_layer=model_cfg['n_layer'], n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'], block_size=model_cfg['block_size'],
        bias=model_cfg.get('bias', False), vocab_size=vocab_size,
        dropout=model_cfg.get('dropout', 0.1),
    )
    model = GPT(gpt_cfg).to(device)
    if train_cfg.get('compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=train_cfg['weight_decay'],
        learning_rate=train_cfg['learning_rate'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
        device_type=device,
    )

    # Gradual curriculum with 500-step ramp (V_G recovery timescale from Agent-C)
    sampler = GradualCurriculumSampler(level_data, ramp_steps=500)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_iters     = train_cfg['max_iters']
    eval_interval = train_cfg['eval_interval']
    eval_iters    = train_cfg['eval_iters']
    block_size    = model_cfg['block_size']
    batch_size    = train_cfg['batch_size']
    grad_accum    = train_cfg.get('gradient_accumulation_steps', 1)
    ckpt_every    = train_cfg.get('checkpoint_every', 1000)
    steps_per_lvl = curric_cfg.get('steps_per_level', 2500)

    metrics_log = []
    weight_log  = []   # track ramp weights over time
    t0 = time.time()

    print(f'Agent-G training: gradual ramp curriculum (no Gnosis)')
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params | Device: {device}')

    for iter_num in range(max_iters + 1):
        sampler.current_step = iter_num

        if iter_num > 0 and iter_num % steps_per_lvl == 0:
            sampler.advance()

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups: pg['lr'] = lr

        if iter_num % eval_interval == 0:
            level_losses = estimate_loss_per_level(
                model, val_data, block_size, batch_size, eval_iters, device)
            weights = sampler.get_weights()

            metrics = {
                'iter': iter_num,
                'current_level': sampler.current_level,
                'level_losses': level_losses,
                'mix_weights': {k: round(v, 3) for k, v in weights.items()},
                'lr': lr,
                'elapsed_s': round(time.time() - t0, 1),
            }
            metrics_log.append(metrics)
            print(f"step {iter_num:5d} L{sampler.current_level} | "
                  + " | ".join(f"L{k}: {v:.4f}" for k, v in level_losses.items())
                  + f" | weights {weights}")

            with open(out_dir / 'metrics.json', 'w') as f:
                json.dump(metrics_log, f, indent=2)

        if iter_num == max_iters: break

        if iter_num > 0 and iter_num % ckpt_every == 0:
            ckpt = {'iter': iter_num, 'current_level': sampler.current_level,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'config': gpt_cfg.__dict__}
            torch.save(ckpt, out_dir / f'ckpt_{iter_num:06d}.pt')

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = sampler.sample(block_size, batch_size, device)
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        optimizer.step()

    torch.save({'iter': max_iters, 'model_state': model.state_dict(),
                'config': gpt_cfg.__dict__}, out_dir / 'agent-g-final.pt')
    print(f'\nAgent-G complete. Final checkpoint: {out_dir}/agent-g-final.pt')


if __name__ == '__main__':
    import random
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='backbone/config.yaml')
    parser.add_argument('--data_dir', default='backbone/data/')
    parser.add_argument('--out_dir',  default='results/agent-g/')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    import yaml
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)
    train(load_config(args.config), args.data_dir, args.out_dir)

