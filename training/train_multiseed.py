#!/usr/bin/env python3
"""
Multi-seed statistical validation for CRS-1.
Runs selected agent(s) across N random seeds and reports
mean ± SD of L1 final loss at step 10000.

Usage:
    # Original comparison (Agent-C vs Agent-N-C):
    python training/train_multiseed.py --agent c --agent n

    # Add gradual ramp:
    python training/train_multiseed.py --agent g

    # Add static-mix baseline:
    python training/train_multiseed.py --agent static

    # All agents:
    python training/train_multiseed.py --agent c --agent n --agent g --agent static

    # Single agent shorthand:
    python training/train_multiseed.py --agent g

Output (stdout + results/multiseed/summary.json)
"""

import subprocess, sys, json, argparse, math
from pathlib import Path


SEED_LIST = [42, 123, 456, 789, 1337]

AGENT_CONFIG = {
    'c': {
        'script': 'training/train_agent_c.py',
        'label': 'Agent-C (hard, no gate)',
        'key': 'agent_c',
        'extra_args': [],
    },
    'n': {
        'script': 'training/train_agent_n.py',
        'label': 'Agent-N-C (hard + gate)',
        'key': 'agent_n_c',
        'extra_args': ['--curriculum', 'hard'],
    },
    'g': {
        'script': 'training/train_agent_g.py',
        'label': 'Agent-G (gradual, no gate)',
        'key': 'agent_g',
        'extra_args': [],
    },
    'static': {
        'script': 'training/train_agent_static.py',
        'label': 'Agent-Static-Mix (fixed 50/50)',
        'key': 'agent_static',
        'extra_args': [],
    },
}


def run_agent(script: str, seed: int, out_dir: str, extra_args: list,
              config: str, data_dir: str) -> float:
    """Run a training script with given seed, return L1 final loss."""
    cmd = [
        sys.executable, script,
        '--seed', str(seed),
        '--config', config,
        '--data_dir', data_dir,
        '--out_dir', out_dir,
    ] + extra_args

    print(f'  Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR (seed={seed}):\n{result.stderr[-500:]}')
        return float('nan')

    metrics_path = Path(out_dir) / 'metrics.json'
    if not metrics_path.exists():
        print(f'  metrics.json not found at {metrics_path}')
        return float('nan')

    with open(metrics_path) as f:
        metrics = json.load(f)

    if not metrics:
        return float('nan')

    last = metrics[-1]
    level_losses = last.get('level_losses', {})
    l1_loss = level_losses.get(1, level_losses.get('1', float('nan')))
    return float(l1_loss)


def mean_std(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float('nan'), float('nan')
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, float('nan')
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(var)


def t_stat(mu1, std1, mu2, std2, n):
    """Welch t-statistic."""
    if any(math.isnan(x) for x in [mu1, std1, mu2, std2]):
        return float('nan')
    se = math.sqrt((std1 ** 2 / n) + (std2 ** 2 / n))
    if se == 0:
        return float('inf') if mu1 != mu2 else 0.0
    return abs(mu1 - mu2) / se


def main():
    parser = argparse.ArgumentParser(description='Multi-seed CRS-1 statistical validation')
    parser.add_argument('--config',   default='backbone/config.yaml')
    parser.add_argument('--data_dir', default='backbone/data/')
    parser.add_argument('--out_dir',  default='results/multiseed/')
    parser.add_argument('--seeds',    type=int, default=5,
                        help='Number of seeds (uses first N from fixed list)')
    parser.add_argument('--alpha',    type=float, default=0.02,
                        help='Gnosis viscosity for Agent-N-C (default 0.02)')
    parser.add_argument('--agent', action='append', dest='agents',
                        choices=['c', 'n', 'g', 'static'],
                        help='Agent(s) to run. Repeat for multiple. '
                             'Default: c n. Choices: c, n, g, static')
    args = parser.parse_args()

    # Default: run Agent-C and Agent-N-C (original comparison)
    agents = args.agents or ['c', 'n']

    seeds = SEED_LIST[:args.seeds]
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    results = {}

    for agent_key in agents:
        cfg = AGENT_CONFIG[agent_key]
        label = cfg['label']
        script = cfg['script']
        result_key = cfg['key']
        extra = list(cfg['extra_args'])

        # Inject alpha for Agent-N-C
        if agent_key == 'n':
            extra = extra + ['--alpha', str(args.alpha)]

        print(f'\n=== {label} — {len(seeds)} seeds ===')
        losses = []
        for seed in seeds:
            out = str(base_out / f'{result_key}/seed_{seed}/')
            loss = run_agent(script, seed, out, extra, args.config, args.data_dir)
            losses.append(loss)
            print(f'  seed={seed}: L1={loss:.4f}')
        results[result_key] = losses

    # Summary
    print('\n=== SUMMARY ===')
    summary = {'seeds': seeds, 'alpha': args.alpha, 'agents': {}}
    stats = {}

    for agent_key in agents:
        rkey = AGENT_CONFIG[agent_key]['key']
        label = AGENT_CONFIG[agent_key]['label']
        vals = results.get(rkey, [])
        mu, sd = mean_std(vals)
        stats[rkey] = (mu, sd)
        summary['agents'][rkey] = {
            'label': label,
            'values': vals,
            'mean': mu,
            'std': sd,
        }
        print(f'{label:40s}  L1: {mu:.4f} ± {sd:.4f}  '
              f'(raw: {[round(v, 4) for v in vals]})')

    # Pairwise t-tests between all agent pairs
    if len(agents) >= 2:
        print('\n--- Pairwise t-statistics ---')
        agent_keys = [AGENT_CONFIG[a]['key'] for a in agents]
        for i in range(len(agent_keys)):
            for j in range(i + 1, len(agent_keys)):
                k1, k2 = agent_keys[i], agent_keys[j]
                l1, l2 = AGENT_CONFIG[agents[i]]['label'], AGENT_CONFIG[agents[j]]['label']
                if k1 in stats and k2 in stats:
                    n = len([v for v in results[k1] if not math.isnan(v)])
                    t = t_stat(stats[k1][0], stats[k1][1],
                               stats[k2][0], stats[k2][1], max(n, 1))
                    flag = '(p<0.0001 likely)' if t > 10 else ('(p<0.05 likely)' if t > 2 else '(not significant)')
                    print(f'  {k1} vs {k2}: t={t:.2f} {flag}')
                    summary['agents'][f't_{k1}_vs_{k2}'] = t

    out_path = base_out / 'summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFull results saved to: {out_path}')


if __name__ == '__main__':
    main()
