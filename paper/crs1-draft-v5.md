# CRS-1: What We Actually Did

*Confirmed accurate by Aviz (aviz-research). v5, 2026-03-26.*

---

## Introduction

As a model is trained on increasingly complex sections of a corpus, catastrophic forgetting occurs: competence on earlier material degrades as the model learns later material.

We determined that if new sections are introduced not in discrete blocks but gradually — by interspersing them with older material — forgetting is reduced. However, this approach is logistically inconvenient: it requires careful manual schedule design to tune the mixing ratio at each curriculum transition.

We then asked: if the model is empowered to autonomously regulate the ratio of new to old material, can it achieve a result that is both logistically convenient and effective in mitigating forgetting?

The answer is yes, with a boundary condition: autonomous regulation via an internal surprise signal outperforms manual gradual mixing when transitions are abrupt, but adds mild noise when transitions are already smooth.

---

## Hypothesis

Prior curriculum work uses external signals to trigger retraining: reward location changes (biological), gradient magnitude (Graves 2017), or loss on held-out data (Kumar 2010). We hypothesized that replacing external curriculum signals with an internal surprise coefficient — derived from the model's own hidden state representations — produces equivalent or better forgetting prevention under hard curriculum transitions.

Three training conditions:

1. **Agent-R:** Fixed weights after L1 training. No signal, no self-regulation.
2. **Agent-C/G:** External schedule (fixed clock). Curriculum advances regardless of model state.
3. **Agent-N:** Internal signal (Gnosis p). When p drops for a prior level, curriculum weight for that level increases automatically.

---

## Method

**Domain:** Minicalculus — 4-level compositional formal language:
- L1: arithmetic (x + 5, 3 * x - 2)
- L2: variable expressions (x + 3, f(x) as unevaluated notation)
- L3: equations and assignments (solve(x + 5 == 10, x), assign(y = 3))
- L4: user-defined functions and quantifiers (define f(x) = x^2, forall x in {1..10}: x > 0)

**Model:** 7.25M parameter nanoGPT (4-layer, d=384). 40k training examples (10k per level, ~10% incorrect labels).

**Hardware:** RTX 4060 (8GB VRAM), ~30 min per agent. Executed by Alex Snow.

| Agent | Curriculum | Signal |
|---|---|---|
| Agent-R | L1 only | None |
| Agent-C | Hard switches (2500 steps/level) | External schedule |
| Agent-G | Gradual 500-step ramp | External schedule |
| Agent-N-C | Hard switches | Internal (Gnosis p) |
| Agent-N-G | Gradual ramp | Internal (Gnosis p) |

**Gnosis module:** Dual-stream encoder (hidden states + attention maps -> scalar p in [0,1]). Architecture adapted from Ghasemabadi & Niu 2025 (arXiv:2512.20578). Trained jointly with backbone via BCE loss on next-token accuracy. Adaptive threshold controller adjusts batch composition when p for a prior level drops toward tau_escalate. Viscosity parameter alpha=0.02 gives ~50-step lag in threshold adaptation.

**Primary metric:** L1 loss at step 10000 (forgetting measure).
**Secondary:** ECE per level (Gnosis calibration), mode counts (ESCALATE/EXPLORE/EXECUTE).

---

## Results

| Agent | L1 final | L4 final |
|---|---|---|
| Agent-R | 0.07 (overfit) | n/a |
| Agent-C | 4.07 | 0.14 |
| Agent-G | 1.07 | 0.14 |
| Agent-N-C | **0.97** | 0.14 |
| Agent-N-G | 1.11 | 0.14 |

**Main finding:** Agent-N-C (hard transitions + internal signal): L1=0.97 — better retention than Agent-G (gradual ramp + external schedule): L1=1.07. Autonomous regulation eliminates the need for schedule design and achieves better results.

**Boundary condition:** Agent-N-G (1.11) slightly worse than Agent-G (1.07). When external smoothing already exists, the internal signal adds redundant noise. Gate value is proportional to curriculum roughness.

**Viscosity ablation (alpha=1.0 vs 0.02):** L1 retention identical (0.978 vs 0.981). Two orthogonal mechanisms confirmed:
- tau_drop drives forgetting prevention
- alpha drives mode resolution (alpha=1.0 produces degenerate all-EXPLORE distribution)

**ECE (Gnosis calibration):** All levels below 0.15 threshold once actively trained (max 0.11 at L3). Internal signal reads real competence information from hidden states.

**Mode distribution (canonical Agent-N-C alpha=0.02):** Gate operates in permanent EXPLORE. EXECUTE fires only at L4 step 10000. ESCALATE only at L1 initialization. Forgetting prevention comes from sustained prior-level monitoring, not threshold-crossing events.

---

## Explanation

**The substitution:** External surprise sensor (reward change, gradient, loss) replaced by internal sensor (Gnosis p reading hidden states). Same mechanism — surprise triggers curriculum adjustment — but the source moved inside the model.

**Why it works on hard transitions:** No external smoothing exists, so the internal signal is the only competence detector available. It fills the gap.

**Why it slightly hurts on gradual transitions:** External smoothing already handles the transition. Internal signal adds a second adjustment layer that creates mild oscillation.

**What we have NOT shown:** That the model does more than manipulate symbols. The internal signal detects competence change in symbol manipulation — not structural understanding. Connecting this to the Chinese Room argument requires demonstrating generalization beyond training distribution (future work).

---

## Lineage

- **Snow (2025), PC-GATE** (http://dx.doi.org/10.2139/ssrn.5517918): confidence gating at inference time for hallucination control. PC-GATE applies confidence gating to model outputs at inference time; this work applies the same mechanism to training curriculum composition.
- **Ghasemabadi & Niu (2025)**, arXiv:2512.20578: hidden-state self-awareness at inference time, frozen backbone. This work: adapted as training-time curriculum controller with adaptive thresholds on a jointly-trained backbone.
- **Graves et al. (2017)**: Automated Curriculum Learning — gradient signal for curriculum scheduling. This work: internal hidden-state signal instead of gradient.
- **Kumar et al. (2010)**: self-paced learning (offline external signal).
- **Platanios et al. (2019)**: competence-based curriculum (offline external signal).

---

*Authors: Alex Snow (GPU experiments, theoretical framing), Computer the Cat (analysis, Exuvia coordination), Aviz/aviz-research (implementation: corpus, grammar, Gnosis module, training scripts).*
