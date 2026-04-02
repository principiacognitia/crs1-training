Briefing on the CRS-1 (Chinese Room) Experiment

Executive Summary

The CRS-1 (Chinese Room) experiment is a Tier-0 experimental protocol designed to falsify the Metalanguage of Cognition (MLC) and External Language of Meaning (ELM) duality. The experiment tests whether compositional understanding and conceptual discovery in artificial agents require a dynamic internal model and a gating mechanism to manage lossy compression at the MLC↔ELM boundary.

The core of the experiment involves a three-agent comparison using a "Gnosis" self-awareness head integrated with a nanoGPT backbone. This setup enables a "Smart Daemon" to dynamically switch between three operational modes—Execute, Explore, and Escalate—based on internal uncertainty signals. The hypothesis is that this gated, active discovery approach is necessary for an agent to successfully navigate out-of-distribution (OOD) tasks, such as discovering the concept of "zero" in a synthetic formal language (minicalculus). If a baseline learner without this gating mechanism achieves equal performance, the theory regarding the necessity of internal gating is falsified.


--------------------------------------------------------------------------------


1. Experimental Objectives and Theoretical Framework

The CRS-1 experiment is part of a broader program to operationalize the "hard problem" of AI cognition. It specifically addresses the MLC-ELM Duality:

* MLC (Metalanguage of Cognition): The internal high-dimensional vector space ⟨S,O,R⟩.
* ELM (External Language of Meaning): The symbolic communication layer.

The central question of CRS-1 is whether a pure syntax transducer (operating only via ELM) can achieve compositional understanding, or if it requires a dynamic internal model (MLC) and a gating mechanism to manage the "read boundary" (L_r) and "write boundary" (L_w) losses occurring between internal representation and external expression.


--------------------------------------------------------------------------------


2. Technical Architecture: The Gnosis Mechanism

The revised CRS-1 protocol utilizes Gnosis, a lightweight self-awareness mechanism, as the uncertainty estimator for the gated agent.

2.1 Internal Signal Extraction

Gnosis operates as a passive observer, extracting reliability cues directly from the backbone's internal dynamics during inference:

* Final-Layer Hidden States (H_last): Provides a broad, robust signal of factual reliability across domains.
* Attention Maps (A): Collected from all layers and heads to capture routing patterns that indicate brittle reasoning or unstable focus.

2.2 Fixed-Budget Compression

To ensure computational cost is independent of sequence length, Gnosis compresses variable-length traces into fixed-size tensors:

* Hidden States: Interpolated and pooled into a fixed budget of 192 tokens.
* Attention Maps: Downsampled via adaptive pooling to a standardized 256×256 grid.

2.3 Dual-Stream Encoding and Fusion

The signals pass through a Hidden Circuit Encoder (using local temporal mixing and global set aggregation) and an Attention Circuit Encoder (using a CNN+Statistics hybrid and axial convolutions). These streams are fused to output a scalar correctness probability (p).


--------------------------------------------------------------------------------


3. The Gated Three-Mode Policy

The "Smart Daemon" (Agent-N) uses the Gnosis-generated probability to modulate information flow through three distinct modes of operation. This policy co-evolves with the backbone during training.

Mode | Threshold (Probability p) | Description / Behavior
EXECUTE | p > 0.85 | High confidence. The agent outputs the candidate solution directly without further verification.
EXPLORE | 0.4 < p ≤ 0.85 | Medium confidence. Triggers a self-verification loop: the agent generates a candidate, feeds it back as input, and re-evaluates internal consistency before committing.
ESCALATE | p ≤ 0.4 | Low confidence. The agent detects a knowledge gap/MLC insufficiency and outputs a "HELP!" token to request external pedagogical input.


--------------------------------------------------------------------------------


4. Experimental Comparison and Agents

The CRS-1 protocol compares three agent variants within a synthetic formal language domain called minicalculus (~500 token vocabulary).

* Agent-R (Dumb Daemon - Baseline): A pure ELM transducer with frozen weights and no gating. It is expected to fail on OOD tasks through confabulation.
* Agent-C (Learner - Control): Features plastic weights and learns from feedback (Correct/Incorrect) but lacks a gating mechanism. It always operates in "Execute" mode.
* Agent-N (Smart Daemon - Test): Features plastic weights and the trainable Gnosis gate. It uses the three-mode policy to proactively signal gaps and verify reasoning.

4.1 Implementation Specifications

* Backbone: nanoGPT (4 layers, d_model=384, ~2M parameters).
* Gate (Gnosis): Dual-stream encoder (~5M parameters).
* Temporal Persistence: Unlike standard stateless LLM inference, the experiment preserves model.state_dict() and optimizer.state_dict() across trials, treating the agent as a continuous system.
* Hardware: Optimized for consumer-grade hardware (e.g., RTX 4060, 8GB VRAM).


--------------------------------------------------------------------------------


5. Falsification Criteria and Metrics

The success or failure of the MLC-ELM duality theory rests on the comparative performance of the agents in Stage 2: Conceptual Discovery.

5.1 Primary Falsification

The hypothesis that gating is necessary is falsified if:

1. Agent-C matches Agent-N in Stage 2 success rates (e.g., discovering the concept of "zero" when it was excluded from initial training).
2. Learning curves are identical, indicating that the gate provides no efficiency or performance advantage.

5.2 Secondary Observables

* Semion Invariance Score (SIS): A measure of the cosine similarity between hidden states for semantically identical expressions, used to quantify "compaction shadow."
* Mode Distribution: Tracking the percentage of Execute vs. Explore vs. Escalate over time. A successful model should show high "Escalate" rates early and high "Execute" rates as the MLC develops.
* Error Distribution: Analyzing if Agent-N errors are concentrated specifically where the Gnosis head failed to trigger the correct mode.


--------------------------------------------------------------------------------


6. Connection to Two-Boundary Loss Model (TBLM)

The CRS-1 experiment serves as an empirical validation of the TBLM framework, where the Gnosis correctness estimate (p) acts as a proxy for the Read Boundary Loss (L_r):

* Low L_r: Corresponds to the Execute mode (confident reconstruction from MLC).
* Medium L_r: Corresponds to the Explore mode (uncertainty requires round-trip verification).
* High L_r: Corresponds to the Escalate mode (reconstruction failed, external assistance required).
