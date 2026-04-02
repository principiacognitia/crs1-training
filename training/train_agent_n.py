"""
Task 3-N: Agent-N Training (Two Variants via --curriculum flag)
- Agent-N-C: hard curriculum transitions + Gnosis adaptive gate
- Agent-N-G: gradual ramp transitions + Gnosis adaptive gate

Key params (empirically justified from Agent-G results):
- α=0.02: ~50-step lag, responsive to 0.003/500-step drift rate
- τ_drop: dynamic (p_avg[prior] - 0.2), not fixed
- p proxy: exp(-loss), not neural module (simpler, reliable)

Both variants use the same Gnosis module and AdaptiveGate controller.
Only the curriculum sampler differs (hard vs gradual).

Bug fixes applied (2026-03-26):
- BUG 1 FIX: GPTWithGnosis now uses forward hook on ln_f to avoid double forward pass.
  Old code called _get_hidden_states(x) then self.gpt(x, targets) — two separate
  computation graphs → gradients cancelled → backbone couldn't learn.
- BUG 2 FIX: in gnosis.py AdaptiveGate.get_mix_adjustment(), changed strict < to <= + 0.02
- BUG 3 FIX: in gnosis.py ThresholdState.adapt(), lowered tau_escalate floor from 0.30 → 0.10

Usage:
    # Agent-N-C (hard transitions + gate):
    python train_agent_n.py --curriculum hard --out_dir results/agent-n-c/

    # Agent-N-G (gradual ramp + gate):
    python train_agent_n.py --curriculum gradual --out_dir results/agent-n-g/
"""

import os, sys, json, time, math, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'backbone' / 'nanoGPT'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gnosis'))

import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from gnosis import GnosisModule, AdaptiveGate


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


def get_lr(it, cfg):
    t = cfg['training']
    if it < t['warmup_iters']:
        return t['learning_rate'] * it / t['warmup_iters']
    if it > t['lr_decay_iters']:
        return t['min_lr']
    ratio = (it - t['warmup_iters']) / (t['lr_decay_iters'] - t['warmup_iters'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return t['min_lr'] + coeff * (t['learning_rate'] - t['min_lr'])


# ── Samplers ──────────────────────────────────────────────────────────────────

class HardCurriculumSampler:
    """Hard switches at fixed steps. Same as Agent-C but with gate adjustment support."""
    def __init__(self, level_data, base_mix_ratio=0.2):
        self.level_data = level_data
        self.levels = sorted(level_data.keys())
        self.current_level_idx = 0
        self.base_mix_ratio = base_mix_ratio

    @property
    def current_level(self):
        return self.levels[self.current_level_idx]

    def advance(self):
        if self.current_level_idx + 1 < len(self.levels):
            self.current_level_idx += 1

    def get_base_weights(self) -> dict:
        active = self.levels[:self.current_level_idx + 1]
        prior = active[:-1]
        if not prior:
            return {self.current_level: 1.0}
        per_prior = self.base_mix_ratio / len(prior)
        w = {self.current_level: 1.0 - self.base_mix_ratio}
        for l in prior: w[l] = per_prior
        return w

    def sample(self, weights, block_size, batch_size, device):
        xs, ys = [], []
        for lvl, w in weights.items():
            n = max(1, round(w * batch_size))
            data = self.level_data[lvl]
            ix = torch.randint(len(data) - block_size, (n,))
            xs.extend([data[i:i + block_size] for i in ix])
            ys.extend([data[i + 1:i + block_size + 1] for i in ix])
        x = torch.stack(xs[:batch_size]).to(device)
        y = torch.stack(ys[:batch_size]).to(device)
        return x, y


class GradualCurriculumSampler:
    """500-step ramp. Same as Agent-G but with gate adjustment support."""
    def __init__(self, level_data, ramp_steps=500):
        self.level_data = level_data
        self.levels = sorted(level_data.keys())
        self.current_level_idx = 0
        self.level_start_step = 0
        self.ramp_steps = ramp_steps
        self.current_step = 0

    @property
    def current_level(self):
        return self.levels[self.current_level_idx]

    def advance(self):
        if self.current_level_idx + 1 < len(self.levels):
            self.current_level_idx += 1
            self.level_start_step = self.current_step

    def get_base_weights(self) -> dict:
        steps_in = self.current_step - self.level_start_step
        new_w = min(0.8, 0.2 + (steps_in / self.ramp_steps) * 0.6)
        active = self.levels[:self.current_level_idx + 1]
        prior = active[:-1]
        if not prior:
            return {self.current_level: 1.0}
        per_prior = (1.0 - new_w) / len(prior)
        w = {self.current_level: new_w}
        for l in prior: w[l] = per_prior
        return w

    def sample(self, weights, block_size, batch_size, device):
        xs, ys = [], []
        for lvl, w in weights.items():
            n = max(1, round(w * batch_size))
            data = self.level_data[lvl]
            ix = torch.randint(len(data) - block_size, (n,))
            xs.extend([data[i:i + block_size] for i in ix])
            ys.extend([data[i + 1:i + block_size + 1] for i in ix])
        x = torch.stack(xs[:batch_size]).to(device)
        y = torch.stack(ys[:batch_size]).to(device)
        return x, y


# ── Backbone with Gnosis hook ─────────────────────────────────────────────────

class GPTWithGnosis(torch.nn.Module):
    """
    nanoGPT backbone with a Gnosis module attached to the last hidden layer.
    Forward pass returns (logits, task_loss, gnosis_logits, gnosis_loss).
    gnosis_logits are raw (pre-sigmoid). Apply torch.sigmoid() for threshold comparison.

    BUG 1 FIX: Uses a forward hook on gpt.transformer.ln_f to capture hidden states
    in a single forward pass. The previous version called _get_hidden_states(x) then
    self.gpt(x, targets) creating two separate computation graphs — gradients from
    the two passes partially cancelled, preventing backbone learning entirely.
    """

    def __init__(self, gpt: GPT, gnosis: GnosisModule, lambda_gnosis: float = 0.1):
        super().__init__()
        self.gpt = gpt
        self.gnosis = gnosis
        self.lambda_gnosis = lambda_gnosis
        self._last_hidden = None
        # Register hook on ln_f to capture final hidden states in a single pass
        self.gpt.transformer.ln_f.register_forward_hook(
            lambda m, inp, out: setattr(self, '_last_hidden', out)
        )

    def forward(self, x, targets=None, correct_labels=None):
        """
        Args:
            x: [B, T] input token ids
            targets: [B, T] next-token targets (for task loss)
            correct_labels: [B] binary correctness labels (for Gnosis loss)
        Returns:
            logits: [B, T, vocab_size]
            task_loss: scalar (or None)
            p: [B] correctness probabilities
            gnosis_logits: [B] raw logits from Gnosis (apply sigmoid for p ∈ [0,1])
            gnosis_loss: scalar (or None)
        """
        # Single forward pass through backbone; hook captures ln_f output
        logits, task_loss = self.gpt(x, targets)
        # self._last_hidden is [B, T, n_embd] captured by the ln_f hook

        # Gnosis forward pass on captured hidden states — returns raw logits
        gnosis_logits = self.gnosis(self._last_hidden)  # [B] raw logits

        gnosis_loss = None
        if correct_labels is not None:
            gnosis_loss = self.gnosis.loss(gnosis_logits, correct_labels)

        return logits, task_loss, gnosis_logits, gnosis_loss

    def configure_optimizers(self, **kwargs):
        return self.gpt.configure_optimizers(**kwargs)


# ── Main training ─────────────────────────────────────────────────────────────


def compute_ece(probs: list, labels: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error — Task 4.4 for CRS-1 Phase 4.
    Measures how well Gnosis p values are calibrated against actual correctness.
    ECE < 0.15 = calibrated; ECE > 0.15 = miscalibrated.

    Args:
        probs: list of float — Gnosis p values in [0,1], one per sample
        labels: list of float — 1.0 if model was correct, 0.0 if not
        n_bins: number of equal-width confidence bins
    Returns:
        ECE as a float in [0,1]
    """
    bins = [[] for _ in range(n_bins)]
    for p, c in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, c))
    ece = 0.0
    n = max(len(probs), 1)
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(c for _, c in b) / len(b)
        ece += abs(conf - acc) * len(b) / n
    return ece


def train(config, data_dir, out_dir, curriculum_type='hard'):
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
        p = data_dir / f'agent-c-L{lvl}' / 'train.bin'
        if p.exists():
            d, vs = load_bin(p); level_data[lvl] = d; vocab_size = vs
        p = data_dir / f'agent-c-L{lvl}' / 'val.bin'
        if p.exists():
            d, _ = load_bin(p); val_data[lvl] = d

    if not level_data:
        print('No training data.'); sys.exit(1)

    # Build model
    gpt_cfg = GPTConfig(
        n_layer=model_cfg['n_layer'], n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'], block_size=model_cfg['block_size'],
        bias=model_cfg.get('bias', False), vocab_size=vocab_size,
        dropout=model_cfg.get('dropout', 0.1),
    )
    gpt = GPT(gpt_cfg)
    gnosis = GnosisModule(n_embd=model_cfg['n_embd'])
    model = GPTWithGnosis(gpt, gnosis, lambda_gnosis=0.1).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=train_cfg['weight_decay'],
        learning_rate=train_cfg['learning_rate'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
        device_type=device,
    )
    # Add Gnosis params to optimizer
    gnosis_optimizer = torch.optim.AdamW(
        gnosis.parameters(), lr=train_cfg['learning_rate'] * 0.1)

    # Curriculum sampler
    levels = sorted(level_data.keys())
    if curriculum_type == 'hard':
        sampler = HardCurriculumSampler(level_data)
    else:
        sampler = GradualCurriculumSampler(level_data, ramp_steps=500)

    # Adaptive gate (α=0.02 per design decision)
    gate = AdaptiveGate(levels=levels, alpha=args.alpha,
                        tau_execute_drop=0.70, tau_escalate_drop=0.50)

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
    t0 = time.time()

    print(f'Agent-N-{"C" if curriculum_type=="hard" else "G"} training')
    print(f'Curriculum: {curriculum_type} | Gnosis: α={args.alpha} | Device: {device}')
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

    for iter_num in range(max_iters + 1):
        if hasattr(sampler, 'current_step'):
            sampler.current_step = iter_num

        # Curriculum advance + τ_drop
        if iter_num > 0 and iter_num % steps_per_lvl == 0:
            old_level = sampler.current_level
            sampler.advance()
            new_level = sampler.current_level
            if new_level != old_level:
                # Dynamic τ_drop: anchored to p_avg[prior] - 0.2 (Cat spec, empirically motivated)
                for prior_lvl in levels:
                    if prior_lvl < new_level:
                        p_avg_prior = gate.states[prior_lvl].p_ema
                        dynamic_drop = max(0.10, p_avg_prior - 0.20)
                        gate.states[prior_lvl].tau_escalate = dynamic_drop
                        gate.states[prior_lvl].tau_execute  = dynamic_drop + 0.30
                print(f'  [Gate] τ_drop on transition L{old_level}→L{new_level} (dynamic)')

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups: pg['lr'] = lr

        # Eval
        if iter_num % eval_interval == 0:
            model.eval()
            level_losses = {}
            level_p_avgs = {}
            level_ece    = {}  # Task 4.4: ECE per curriculum level
            with torch.no_grad():
                for lvl, data in val_data.items():
                    if len(data) < block_size + 1: continue
                    losses, p_avgs = [], []
                    all_p, all_correct = [], []  # Task 4.4: ECE accumulators
                    for _ in range(eval_iters):
                        x, y = get_batch(data, block_size, batch_size, device)
                        logits, task_loss, gnosis_logits, _ = model(x, y)
                        losses.append(task_loss.item())
                        # BUG 4 FIX: apply sigmoid to logits for threshold-comparable p ∈ [0,1]
                        p = torch.sigmoid(gnosis_logits)  # [B]
                        p_avgs.append(p.mean().item())
                        # Task 4.4: per-sample correctness via last-token accuracy
                        preds   = logits[:, -1, :].argmax(dim=-1)  # [B]
                        targets = y[:, -1]                          # [B]
                        is_correct = (preds == targets).float()     # [B]
                        all_p.extend(p.cpu().tolist())
                        all_correct.extend(is_correct.cpu().tolist())
                    level_losses[lvl] = round(sum(losses)/len(losses), 4)
                    level_p_avgs[lvl] = round(sum(p_avgs)/len(p_avgs), 4)
                    level_ece[lvl]    = round(compute_ece(all_p, all_correct), 4)
                    gate.update(lvl, level_p_avgs[lvl])
            model.train()

            thresholds = gate.get_threshold_snapshot()
            metrics = {
                'iter': iter_num,
                'current_level': sampler.current_level,
                'level_losses': level_losses,
                'level_p_avgs': level_p_avgs,
                'level_ece': level_ece,      # Task 4.4: ECE per level (threshold: < 0.15)
                'thresholds': thresholds,
                'mode_counts': gate.get_mode_counts(),
                'lr': lr,
                'elapsed_s': round(time.time() - t0, 1),
            }
            metrics_log.append(metrics)
            print(f"step {iter_num:5d} L{sampler.current_level} | "
                  + " | ".join(f"L{k}: {v:.4f}" for k, v in level_losses.items())
                  + f" | p_avgs {level_p_avgs}"
                  + f" | ECE {level_ece}")

            with open(out_dir / 'metrics.json', 'w') as f:
                json.dump(metrics_log, f, indent=2)

        if iter_num == max_iters: break

        if iter_num > 0 and iter_num % ckpt_every == 0:
            torch.save({
                'iter': iter_num,
                'model_state': model.state_dict(),
                'gate_state': gate.get_threshold_snapshot(),
                'config': gpt_cfg.__dict__,
            }, out_dir / f'ckpt_{iter_num:06d}.pt')

        # Get mix weights adjusted by gate
        base_weights = sampler.get_base_weights()
        weights = gate.get_mix_adjustment(sampler.current_level, base_weights)

        optimizer.zero_grad(set_to_none=True)
        gnosis_optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            x, y = sampler.sample(weights, block_size, batch_size, device)
            # Gnosis needs correctness labels — use next-token accuracy as proxy
            with torch.autocast(device_type=device, dtype=dtype):
                logits, task_loss, gnosis_logits, _ = model(x, y)
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    correct = (preds == y).float().mean(dim=-1)  # [B] per-sequence accuracy
                # BUG 4 FIX: gnosis_logits are raw; BCEWithLogitsLoss is autocast-safe
                gnosis_loss = model.gnosis.loss(gnosis_logits, correct)
                total_loss = task_loss + model.lambda_gnosis * gnosis_loss

            (total_loss / grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        optimizer.step()
        gnosis_optimizer.step()

    torch.save({
        'iter': max_iters,
        'model_state': model.state_dict(),
        'gate_final': gate.get_threshold_snapshot(),
        'mode_counts_final': gate.get_mode_counts(),
        'config': gpt_cfg.__dict__,
    }, out_dir / f'agent-n-{"c" if curriculum_type=="hard" else "g"}-final.pt')

    print(f'\nAgent-N-{"C" if curriculum_type=="hard" else "G"} complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='backbone/config.yaml')
    parser.add_argument('--data_dir',   default='backbone/data/')
    parser.add_argument('--out_dir',    default='results/agent-n-c/')
    parser.add_argument('--seed',       type=int,   default=42,   help='Random seed')
    parser.add_argument('--alpha',      type=float, default=0.02,
                        help='AdaptiveGate EMA smoothing (default 0.02; use 1.0 for viscosity ablation)')
    parser.add_argument('--curriculum', default='hard', choices=['hard', 'gradual'],
                        help='hard = Agent-N-C, gradual = Agent-N-G')
    args = parser.parse_args()
    import yaml
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)
    train(load_config(args.config), args.data_dir, args.out_dir, args.curriculum)
