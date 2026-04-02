# Gnosis: Decoding Internal Model Traces for Real-Time LLM Verification

**Paper:** arXiv:2512.20578v1  
**Authors:** [from paper]  
**Relevance to CRS-1:** Trainable 5M-parameter uncertainty estimator that operationalizes gating mechanism

## Key Specifications

- **Parameter count:** ~5M (1000x smaller than 8B reward models)
- **Training cost:** ~$25 for 20B model
- **Latency overhead:** 25ms per forward pass
- **Performance:** 0.95-0.96 AUROC on Math tasks (vs 0.81-0.90 for 8B reward models)
- **Architecture:** Dual-stream (Hidden Circuit Encoder + Attention Circuit Encoder)

## Fixed-Budget Compression

- **Hidden states:** K_hid=192 (adaptive pool from final layer)
- **Attention maps:** k=256 (downsample from all layers to 256x256 grid)

## Training Objective

Binary Cross-Entropy (BCE) on correctness labels:
```
L_Gnosis = BCE(p, correctness_label)
```

Where p in [0,1] is scalar correctness probability output.

## CRS-1 Integration

**Joint training:**
```
L = L_task + lambda * L_Gnosis
```

**Emergent gating policy:**
- High p (>0.85) → EXECUTE mode
- Medium p (0.4-0.85) → EXPLORE mode  
- Low p (<0.4) → ESCALATE mode

Thresholds emerge from validation set optimization, not manual tuning.

**Key innovation:** Gate policy co-evolves with backbone capability. Early training: ESCALATE frequently. Late training: EXECUTE dominantly.

## Reference

Full paper: https://arxiv.org/html/2512.20578v1