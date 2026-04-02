# Follow-up Questions for External Reviewer (Gemini 3.1 Pro)

*To be pasted into Gemini 3.1 Pro with the original review as context.*

---

## Context

We have read your review of CRS-1. The critique about statistical significance and toy scale is accepted. Before implementing refinements, we want your assessment of two proposed extensions.

---

## Proposed Extension 1: Scale Experiment

Run the same 2x2 factorial (Agent-C vs Agent-N-C) on a medium-scale nanoGPT:
- 8 layers, 12 heads, d=512, ~55M params (vs current 7.25M)
- Same Minicalculus corpus
- Compare learning curves to current results

**Question:** Does this address your scalability concern sufficiently? If the interaction effect (gate rescues hard transitions) holds at 55M params, is that adequate evidence of scalability for a workshop paper? Or do you consider 55M still toy scale?

---

## Proposed Extension 2: Natural Language Corpus (Multilingual)

Replace Minicalculus with 4 Project Gutenberg books in 4 languages:
- L1: English
- L2: French (Romance, close to English)
- L3: German (Germanic, different word order)
- L4: Finnish (Uralic — agglutinative, 15 grammatical cases, entirely different structure)

Use a language model to generate 200 factual QA pairs per language from each book. Train the same agents (Agent-C vs Agent-N-C). 

**Success criterion:** At step 10000, model answers English questions in English AND Finnish questions in Finnish. L1 retention = English QA accuracy maintained after Finnish training.

**Question:** Does this adequately address the training set concern? Is factual QA across typologically distant languages a sufficient analog to the compositional difficulty ordering in Minicalculus? What would make this test convincing or unconvincing?

---

## Combined Question

Do both extensions together (scale + multilingual) move the paper from "unpublishable" to "workshop-ready"? Or is there a more targeted intervention you would prioritize?
