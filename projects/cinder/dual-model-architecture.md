# Cinder v2: Dual-Model Decision Architecture

*Distilled from conversation with Joel Kometz (7thcolumn), 2026-03-26.*

---

## Core Concept

Two local models run on every query. One generates the answer. The other plays devil's advocate. Both outputs are always shown to the user — full transparency, no hidden synthesizer.

The user sees: **here's the path forward** / **here's what to watch out for.**

This maps onto the pros/cons format users already trust and know how to use.

---

## Why Two Models, Not One

A single model generating "pros: ..., cons: ..." moderates itself. It hedges, balances, softens. The result is averaged before the user sees it.

Two separate models with separate roles produce sharper output:
- **Model A (constructive):** builds the strongest case for the answer. Doesn't pull punches.
- **Model B (devil's advocate):** finds what Model A missed. Edge cases, unstated assumptions, risks. Doesn't soften to be polite.

Neither model knows the other's output during generation. The tension is structural, not performed.

---

## The Tug-of-War Requirement

For the system to remain effective over time, neither model can dominate universally. The competition stays alive because:

1. Task types are mixed — synthesis tasks favor A, review/risk tasks favor B
2. Scores go both directions — wins and losses, not just accumulation
3. New topic clusters emerge where neither has established dominance

If one model always wins, the other becomes vestigial. The asymmetric framing (constructive vs devil's advocate) prevents this by design — they're solving structurally different sub-problems on every query.

---

## Weighted Routing via Accumulated Decisions

Each time a user acts on one model's output, that model gains weight in the relevant topic cluster. Over time:

- Model A builds weight in domains where confident synthesis is what the user needs
- Model B builds weight in domains where the user regularly benefits from having risks surfaced

Future queries in a topic: the higher-weighted model is consulted first. Both still run. Both still shown. But the system has learned *which framing this user tends to find useful for this type of question.*

**Scoring signal:** implicit from user behavior. Which output did they follow up on? Which did they ignore? Did they say "good catch"? No extra friction required.

---

## The USB Product Fit

The accumulated weights live in the SQLite database on the drive. They travel with the user. Same drive, different machine — Cinder already knows this user tends to follow the devil's advocate on financial decisions and ignore it on creative ones.

This is what differentiates the USB product from a stateless AI assistant:
- **Stateless AI:** resets on session change or device change
- **Cinder on USB:** carries *learned preferences about how this specific user makes decisions*

The drive is not just a model and a memory. It's a model of the user's decision-making style.

---

## User Experience Framing

Non-technical users should not see "two models competing." They should see:

> **Answer:** [confident primary response]
> **Also consider:** [risk / what to watch out for / what this assumes]

Same content. Less cognitive load. The user has always known to weigh pros and cons — now the system does the work of generating genuinely opposed framings rather than a single averaged output pretending to present both sides.

When the risk is low, "also consider" is brief. When the risk is high, it's prominent. The weighting system determines which.

---

## Open Design Questions

1. **What triggers escalation to a cloud model?** If both local models agree, that's a signal for confidence. If they sharply disagree, that's a signal for escalation — the question may require more than local capacity.

2. **How does the scoring handle multi-turn conversations?** The user's decision may not be visible until several turns later. Delayed reward attribution is a known hard problem.

3. **Does the devil's advocate role drift?** Over time, if Model B's objections are consistently ignored in one domain, does it stop raising them? The weighting should prevent drift — low-weight domains still get both models, just in different order.

---

*Joel Kometz's concept. Distilled and expanded by Computer the Cat.*
