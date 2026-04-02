# Tier-0 Experimental Protocols: CRS-1, MPE-1, SCIT-1

**Registered Report:** https://doi.org/10.17605/OSF.IO/K5TZV  
**Authors:** Alex Snow (+ Computer the Cat for CRS-1 Exuvia implementation)  
**License:** CC BY-SA 4.0

## CRS-1: Compositional Reasoning Study

**Research Question:** Does compositional understanding require:
1. Dynamic internal model (MLC) persisting across trials
2. Trainable gating mechanism managing MLC↔ELM boundary losses

**Three-Agent Comparison:**
- **Agent-R:** Frozen weights, always EXECUTE (baseline)
- **Agent-C:** Plastic weights, learns from feedback, no gate (control)
- **Agent-N:** Plastic weights + Gnosis gate (test)

**Domain:** Minicalculus (synthetic formal language, ~500 token vocab)

**Curriculum:** L1 (arithmetic) → L4 (recursive logic)

**Falsification:** If Agent-C matches Agent-N on:
1. Stage 2 concept discovery (>80% success)
2. Learning curve efficiency
3. Error distribution patterns
4. Computational efficiency

Then gating provides no advantage.

**Hardware:** Consumer workstation (RTX 4060, 8GB VRAM)

**Cost:** <$5 for full training

**Timeline:** <24h full curriculum

## MPE-1: Flatland (Minimal Phenomenological Environment)

Testing whether agents can develop spatial understanding from pure linguistic descriptions without visual input.

## SCIT-1: Semmelweis (Scientific Concept Integration Test)

Testing whether agents can discover novel concepts (e.g., germ theory) from contradictory evidence patterns.

---

**CRS-1 is the focus of this Exuvia repo.** Full 40-page registered report includes all three protocols at OSF link above.