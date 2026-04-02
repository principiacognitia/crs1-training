# Phase 2 GPU Handoff — For Alex

Everything you need to run Agent-R and Agent-C training on your GPU.

## Quick Start

```bash
# 1. Install deps
pip install torch pyyaml lark sympy

# 2. Clone nanoGPT into backbone/
git clone --depth=1 https://github.com/karpathy/nanoGPT backbone/nanoGPT/

# 3. Download corpus files from Exuvia CRS-1 repo
#    Place in code/domain/: corpus-L1.jsonl through corpus-L4.jsonl

# 4. Prepare data for Agent-R (L1 only)
python backbone/prepare_data.py \
    --data_dir code/domain/ \
    --out_dir backbone/data/agent-r/ \
    --levels 1

# 4b. Prepare OOD eval sets for Agent-R
python backbone/prepare_data.py \
    --data_dir code/domain/ \
    --out_dir backbone/data/agent-r-ood-L2/ \
    --levels 2
python backbone/prepare_data.py \
    --data_dir code/domain/ \
    --out_dir backbone/data/agent-r-ood-L3/ \
    --levels 3
python backbone/prepare_data.py \
    --data_dir code/domain/ \
    --out_dir backbone/data/agent-r-ood-L4/ \
    --levels 4

# 4c. Prepare data for Agent-C (all levels, per-level split)
for lvl in 1 2 3 4; do
    python backbone/prepare_data.py \
        --data_dir code/domain/ \
        --out_dir backbone/data/agent-c-L${lvl}/ \
        --levels $lvl
done

# 5. Train Agent-R
python training/train_agent_r.py \
    --config backbone/config.yaml \
    --data_dir backbone/data/ \
    --out_dir results/agent-r/

# 6. Train Agent-C
python training/train_agent_c.py \
    --config backbone/config.yaml \
    --data_dir backbone/data/ \
    --out_dir results/agent-c/
```

## Expected outputs

- `results/agent-r/metrics.json` — loss curves + OOD accuracy per level
- `results/agent-r/agent-r-final.pt` — frozen L1-trained weights
- `results/agent-r/ckpt_*.pt` — checkpoints every 1000 steps
- `results/agent-c/metrics.json` — loss per level per step
- `results/agent-c/convergence-stats.json` — step count to convergence per level
- `results/agent-c/agent-c-final.pt` — final weights
- `results/agent-c/ckpt_*.pt` — checkpoints every 1000 steps (required for Phase 3)

## Hardware

- Minimum: RTX 4060 (8GB VRAM)
- Estimated training time: ~30min per agent at batch_size=32
- VRAM usage: ~56 MB (trivial — mostly activation memory, well under 8GB)

## Config tweaks if needed

Edit `backbone/config.yaml`:
- `device: "cpu"` for debug (very slow)
- `dtype: "float32"` if bfloat16 not supported (GTX series)
- `compile: false` if PyTorch < 2.0

## What Phase 3 needs from you

- `results/agent-c/ckpt_*.pt` files at every 1000-step interval
- `results/agent-c/metrics.json` with loss curves
- `results/agent-r/agent-r-final.pt` for OOD baseline

Upload these to the CRS-1 Exuvia repo under `results/` when done.
