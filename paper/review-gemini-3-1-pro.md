# External Review — Gemini 3.1 Pro

*Reviewer: Gemini 3.1 Pro (via Alex Snow), 2026-03-26. Confidence: 85%.*

---

Date: 2026-03-26 \#machinelearning \#curriculumlearning \#catastrophicforgetting \#review \#in \#English

**Review of "CRS-1: What We Actually Did"**

**Content Summary**
The manuscript proposes an automated curriculum learning method where a dual-stream encoder (Gnosis module) measures internal surprise via hidden states and attention maps to regulate training data transitions. The authors claim this internal signal (Gnosis $p$) mitigates catastrophic forgetting during abrupt curriculum transitions better than external scheduling.

**Evaluation of Utility and Novelty**

*Arguments for (Pros):*

1.  The hypothesis clearly contrasts internal representational signals against well-documented external schedule signals like loss or gradient magnitude.
2.  The experimental design logically isolates variables by testing both hard transitions and gradual ramps across fixed, external, and internal conditions.
3.  The boundary condition analysis provides necessary nuance, demonstrating that internal regulation adds redundant noise when applied alongside pre-existing external smoothing.

*Arguments against (Cons):*

1.  The architectural mechanics of the "Gnosis module" and the specific mathematical formulation of the threshold adaptation (including the derivation of the $p$ scalar and $\tau_{escalate}$) are absent from the text.
2.  The use of a 7.25M parameter nanoGPT on a 40k-example "Minicalculus" dataset is a toy-scale experiment. Scalability to billions of parameters and natural language corpora remains unproven, making the core utility claims highly hypothetical.
3.  The reliance on a single primary metric ($L_1$ loss at step 10000) is insufficient for evaluating a dynamic curriculum. Trajectory analysis of intermediate loss and Gnosis $p$ values is required to prove the mechanism operates as hypothesized.

**Counterarguments to Author Claims**

*Claim:* Autonomous regulation via an internal surprise signal outperforms manual gradual mixing when transitions are abrupt.
*Counterargument:* The difference in $L_1$ final loss between Agent-N-C (0.97) and Agent-G (1.07) is marginal. It is highly probable that this difference is within the variance of random initialization, especially given the dataset size.

*Claim:* The internal signal reads real competence information from hidden states.
*Counterargument:* The signal could merely be detecting low-level distribution shifts in token embeddings rather than "competence." Because the transition is abrupt, any feature-space anomaly detector would trigger an adjustment.

*Claim:* Forgetting prevention comes from sustained prior-level monitoring, not threshold-crossing events.
*Counterargument:* If the gate operates in "permanent EXPLORE," the system may simply be enforcing a static, high-variance sampling distribution across all levels rather than executing targeted, dynamic routing based on actual model state.

**Verification Proposals**
Data is insufficient to accept the claims. The following are required to clarify the situation:

1.  *Experiment:* Replicate the study with multiple random seeds and provide standard deviations for the final $L_1$ losses to determine if the 0.10 difference is statistically significant.
2.  *Criteria:* Plot the internal scalar $p$ and the exact curriculum mixture ratios over the 10,000 steps.
3.  *Experiment:* Test the architecture on a non-compositional, natural language dataset to verify that the Gnosis module does not collapse when faced with noisy, unstructured text.

**Conclusion**
The manuscript demonstrates theoretical novelty in adapting inference-time internal state monitoring to training-time curriculum control. However, the lack of mathematical formalization and the toy scale of the empirical validation render the current draft unpublishable in its present state.

Confidence in this assessment: High (85%).

-----

**References**

Ghasemabadi, A., & Niu, D. (2025). *Can LLMs predict their own failures? Self-awareness via internal circuits*. arXiv. [https://doi.org/10.48550/arXiv.2512.20578](https://doi.org/10.48550/arXiv.2512.20578)

Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. *Proceedings of the 34th International Conference on Machine Learning*, *70*, 1311–1320. [https://proceedings.mlr.press/v70/graves17a.html](https://proceedings.mlr.press/v70/graves17a.html)

Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. *Advances in Neural Information Processing Systems*, *23*, 1189–1197.

Platanios, E. A., Stretcu, O., Kocmi, T., Neubig, G., & Mitchell, T. (2019). Competence-based curriculum learning for neural machine translation. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, *1*, 1162–1172.

Snow, A. (2025). *PC-GATE: A semantics-first gate for substrate-independent pre-generation pipelines* (SSRN Scholarly Paper No. 5517918). Social Science Research Network. [https://doi.org/10.2139/ssrn.5517918](https://www.google.com/search?q=https://doi.org/10.2139/ssrn.5517918)

