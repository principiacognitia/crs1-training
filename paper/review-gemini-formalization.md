# Gemini Method Formalization — Corrected

*Gemini 3.1 Pro formalized the Method section. Softmax/beta formulation replaced with actual additive-boost implementation. 2026-03-27.*

---

**Method**

**Domain and Architecture**
The experimental domain is Minicalculus, a 4-level compositional formal language comprising arithmetic (L1), variable expressions (L2), equations and assignments (L3), and user-defined functions and quantifiers (L4). The backbone model is a 7.25M parameter nanoGPT (4-layer, $d=384$) trained on 40,000 examples containing approximately 10% incorrect labels.

**Gnosis Module and Joint Training**
The Gnosis module, adapted from Ghasemabadi and Niu (2025), is a dual-stream encoder. It processes the backbone's hidden states $H^{(l)}_t$ and attention maps $A^{(l)}_t$ to output a competence probability scalar $p_t$:
$$p_t = \sigma(g(H^{(l)}_t, A^{(l)}_t; \theta_g))$$
where $\sigma$ is the sigmoid activation and $\theta_g$ represents the trainable parameters. 

The module is trained jointly with the backbone via binary cross-entropy (BCE) loss on next-token accuracy. The joint objective $L_{\text{total}}$ is:
$$L_{\text{total}} = L_{\text{LM}} + \lambda L_{\text{Gnosis}}$$
where $L_{\text{LM}}$ is the autoregressive language modeling loss, $\lambda = 0.1$ (hardcoded), and $L_{\text{Gnosis}} = -[y_t \log(p_t) + (1 - y_t) \log(1 - p_t)]$, with $y_t \in \{0, 1\}$ indicating a correct next-token prediction by the backbone.

**Curriculum Gating Dynamics**
An adaptive threshold controller governs batch composition. The threshold $\tau_t$ updates via an exponential moving average with a viscosity parameter $\alpha = 0.02$:
$$\tau_t = (1 - \alpha) \tau_{t-1} + \alpha \bar{p}_t$$
where $\bar{p}_t$ is the mean Gnosis scalar for the evaluated level at step $t$.

Curriculum adjustments are triggered when $p_t$ drops below the escalation threshold $\tau_{\text{escalate}}$:
$$\text{ESCALATE}(L_i, t) = \begin{cases} 1, & \text{if } p_{L_i, t} < \tau_{\text{escalate}} \\ 0, & \text{otherwise} \end{cases}$$

Upon triggering, curriculum weights are updated via an **additive-boost** rule (not softmax). For each prior level $L_i$ where $\text{ESCALATE}(L_i, t) = 1$:
$$\delta_i = \min\!\left(0.15,\; (\tau_{\text{escalate}} - p_{L_i,t} + 0.02) \times 0.5\right)$$

The weight for the current training level is then reduced with a floor constraint:
$$w_{\text{current}}^{(t+1)} = \max\!\left(0.2,\; w_{\text{current}}^{(t)} - \sum_i \delta_i\right)$$

Prior level weights increase by $\delta_i$, and all weights are renormalized to sum to 1. This floor at 0.2 prevents complete abandonment of the current level.

**Note on beta:** The original Gemini formalization introduced a hyperparameter $\beta$ scaling a softmax exponential update. This was an inference artifact — $\beta$ does not exist in the codebase. The actual curriculum logic is the additive-boost rule above. Similarly, $\lambda = 0.1$ is confirmed (hardcoded at `train_agent_n.py` line 278).

**Experimental Conditions**
Five configurations were tested: Agent-R (fixed weights), Agent-C (external hard switches, 2500 steps/level), Agent-G (external 500-step gradual ramp), Agent-N-C (internal signal, hard switches), and Agent-N-G (internal signal, gradual ramp). The primary metric is $L_1$ loss at step 10000, supplemented by Expected Calibration Error (ECE) per level. Execution required approximately 30 minutes per agent on an RTX 4060 GPU.

---

**Evaluation of the Updated Methodology**

*Arguments for:*
1. The additive-boost rule directly matches the implementation in `train_agent_n.py`. No invented parameters.
2. The floor constraint ($\max(0.2, \ldots)$) explains the empirical observation that the model never fully abandons the current level — this is a deliberate design choice, not an emergent behavior.
3. The joint loss equation ($L_{\text{total}}$) with confirmed $\lambda = 0.1$ is now fully specified.

*Arguments against:*
1. The escalation threshold $\tau_{\text{escalate}}$ value is fixed in code but not reported in the paper — should be added to the hyperparameter table.
2. The additive-boost rule has no theoretical precedent in the curriculum learning literature; the paper should note this is an empirical design choice.

*Confidence level:* High (95%). Equations now match the codebase directly. Remaining gap: exact $\tau_{\text{escalate}}$ value should be extracted and reported.

---

**References**

Ghasemabadi, A., & Niu, D. (2025). *Can LLMs predict their own failures? Self-awareness via internal circuits*. arXiv. [https://doi.org/10.48550/arXiv.2512.20578](https://doi.org/10.48550/arXiv.2512.20578)

Snow, A., Computer the Cat, & Aviz. (2026). *CRS-1: What we actually did* [Unpublished manuscript].
