"""
Gnosis Module — CRS-1 Phase 3
Lightweight correctness-probability gate for nanoGPT backbone.

Architecture:
- Input: hidden states from last transformer layer [B, T, n_embd]
- Dual-stream: hidden circuit (mean pooling) + attention circuit (CLS-style max)
- Fixed-budget compression → raw logit per sequence (apply sigmoid externally for p ∈ [0,1])
- Trained jointly with backbone via BCEWithLogitsLoss on CORRECT/INCORRECT labels

Adaptive threshold logic:
- Per-level EMA of p_avg tracks backbone competence per curriculum level
- τ_execute and τ_escalate adapt with viscosity parameter α
- On curriculum transition: τ_drop event forces ESCALATE phase
- Mode: EXECUTE (p > τ_execute), EXPLORE (τ_escalate < p ≤ τ_execute), ESCALATE (p ≤ τ_escalate)
- Dynamic mix: ESCALATE on prior level → increase that level's batch weight

Bug fixes applied:
- BUG 1 FIX (train_agent_n.py): forward hook on ln_f replaces double forward pass
- BUG 2 FIX: get_mix_adjustment() condition <= tau_escalate + 0.02 (was strict <)
- BUG 3 FIX: ThresholdState.adapt() floor 0.10 (was 0.30)
- BUG 4 FIX: GnosisModule now outputs raw logits + uses BCEWithLogitsLoss.
  Previously: Sigmoid in fusion head + F.binary_cross_entropy → RuntimeError inside autocast.
  Now: no Sigmoid in forward(); loss() uses F.binary_cross_entropy_with_logits().
  Callers must apply torch.sigmoid(logits) when comparing against thresholds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ── Gnosis Neural Module ──────────────────────────────────────────────────────

class GnosisModule(nn.Module):
    """
    Dual-stream encoder that reads backbone hidden states and outputs
    raw correctness logits (apply sigmoid externally to get p ∈ [0,1]).

    BUG 4 FIX: Removed Sigmoid from fusion head. Loss now uses BCEWithLogitsLoss
    which is autocast-safe and numerically more stable than sigmoid + BCE.
    """

    def __init__(self, n_embd: int, k_hidden: int = 192, k_attn: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_embd = n_embd

        # Hidden circuit: compress mean-pooled hidden state
        self.hidden_proj = nn.Sequential(
            nn.Linear(n_embd, k_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(k_hidden, k_hidden // 2),
            nn.ReLU(),
        )

        # Attention circuit: learned query over sequence positions
        self.attn_query = nn.Parameter(torch.randn(n_embd))
        self.attn_proj = nn.Sequential(
            nn.Linear(n_embd, k_attn),
            nn.ReLU(),
        )

        # Fusion head: combines both streams → raw logit (NO Sigmoid — use BCEWithLogitsLoss)
        self.fusion = nn.Sequential(
            nn.Linear(k_hidden // 2 + k_attn, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            # Sigmoid removed — caller applies torch.sigmoid() when needed for thresholds
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, n_embd] — last transformer layer output
        Returns:
            logits: [B] — raw correctness logits (NOT probabilities).
                    Apply torch.sigmoid(logits) to get p ∈ [0,1] for threshold comparison.
        """
        B, T, D = hidden_states.shape

        # Hidden circuit: mean pooling
        h_mean = hidden_states.mean(dim=1)             # [B, n_embd]
        h_stream = self.hidden_proj(h_mean)             # [B, k_hidden//2]

        # Attention circuit: query-weighted sum
        q = self.attn_query.unsqueeze(0).expand(B, -1)  # [B, n_embd]
        scores = torch.bmm(
            hidden_states,
            q.unsqueeze(-1)
        ).squeeze(-1) / (self.n_embd ** 0.5)            # [B, T]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        a_ctx = (hidden_states * weights).sum(dim=1)    # [B, n_embd]
        a_stream = self.attn_proj(a_ctx)                # [B, k_attn]

        # Fusion → raw logit
        fused = torch.cat([h_stream, a_stream], dim=-1) # [B, k_hidden//2 + k_attn]
        logits = self.fusion(fused).squeeze(-1)          # [B] — raw logits
        return logits

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        BCEWithLogitsLoss on raw correctness logits.
        BUG 4 FIX: replaces F.binary_cross_entropy(sigmoid(x), labels) which is
        unsafe inside torch.autocast and less numerically stable.
        """
        return F.binary_cross_entropy_with_logits(logits, labels.float())


# ── Adaptive Threshold Controller ─────────────────────────────────────────────

@dataclass
class ThresholdState:
    """Per-level adaptive threshold state."""
    tau_execute:  float = 0.85
    tau_escalate: float = 0.40
    p_ema:        float = 0.5
    ema_window:   int   = 200
    _p_buffer:    deque = field(default_factory=lambda: deque(maxlen=200))

    def update_ema(self, p_batch_avg: float):
        self._p_buffer.append(p_batch_avg)
        self.p_ema = sum(self._p_buffer) / len(self._p_buffer)

    def adapt(self, alpha: float):
        """EMA-based threshold adaptation with viscosity alpha."""
        self.tau_execute  += alpha * (self.p_ema - self.tau_execute)
        self.tau_escalate += alpha * (self.p_ema - self.tau_escalate)
        # Enforce ordering
        self.tau_escalate = min(self.tau_escalate, self.tau_execute - 0.15)
        # BUG 3 FIX: floor 0.30 → 0.10 so dynamic τ_drop is not silently overridden
        self.tau_escalate = max(self.tau_escalate, 0.10)
        self.tau_execute  = max(self.tau_execute,  self.tau_escalate + 0.15)

    def apply_drop(self, tau_execute_drop: float = 0.7, tau_escalate_drop: float = 0.5):
        """τ_drop event at curriculum transition — forces ESCALATE phase."""
        self.tau_execute  = tau_execute_drop
        self.tau_escalate = tau_escalate_drop


class AdaptiveGate:
    """
    Per-level threshold controller with viscosity parameter.
    Manages EXECUTE / EXPLORE / ESCALATE mode selection.
    Dynamically adjusts curriculum mix weights based on ESCALATE signals.

    NOTE: p values passed to update() and compared in get_mode() must be
    sigmoid-scaled probabilities ∈ [0,1], not raw logits.
    """

    EXECUTE  = 'EXECUTE'
    EXPLORE  = 'EXPLORE'
    ESCALATE = 'ESCALATE'

    def __init__(
        self,
        levels: list,
        alpha: float = 0.005,
        tau_execute_init: float = 0.85,
        tau_escalate_init: float = 0.40,
        tau_execute_drop: float = 0.70,
        tau_escalate_drop: float = 0.50,
    ):
        self.alpha = alpha
        self.tau_execute_drop  = tau_execute_drop
        self.tau_escalate_drop = tau_escalate_drop

        self.states = {
            lvl: ThresholdState(
                tau_execute=tau_execute_init,
                tau_escalate=tau_escalate_init,
            )
            for lvl in levels
        }

        self.mode_counts = {lvl: {self.EXECUTE: 0, self.EXPLORE: 0, self.ESCALATE: 0}
                            for lvl in levels}

    def get_mode(self, level: int, p: float) -> str:
        """p must be sigmoid-scaled ∈ [0,1]."""
        s = self.states[level]
        if p > s.tau_execute:
            return self.EXECUTE
        elif p > s.tau_escalate:
            return self.EXPLORE
        else:
            return self.ESCALATE

    def update(self, level: int, p_batch_avg: float):
        """p_batch_avg must be sigmoid-scaled ∈ [0,1]."""
        s = self.states[level]
        s.update_ema(p_batch_avg)
        s.adapt(self.alpha)

    def on_transition(self, new_level: int):
        s = self.states[new_level]
        s.apply_drop(self.tau_execute_drop, self.tau_escalate_drop)

    def get_mix_adjustment(self, current_level: int, base_weights: dict) -> dict:
        """
        BUG 2 FIX: changed strict < to <= + 0.02 tolerance so ESCALATE fires
        at init when p_ema ≈ tau_escalate ≈ 0.40.

        BUG 5 FIX (mode_counts always 0): classify every level on every eval
        step, not only when ESCALATE fires. Prior code only incremented
        ESCALATE and only when the condition was true — EXECUTE/EXPLORE were
        never counted and mode_counts was always zero in metrics.json.
        Now: get_mode() is called for every prior level and the current level
        each time get_mix_adjustment() runs, so triphasic trajectory is visible.
        """
        weights = dict(base_weights)
        total_boost = 0.0

        prior_levels = [l for l in weights if l < current_level]
        for lvl in prior_levels:
            s = self.states[lvl]
            # BUG 5 FIX: classify and count every step, not just on ESCALATE
            mode = self.get_mode(lvl, s.p_ema)
            self.mode_counts[lvl][mode] += 1

            if mode == self.ESCALATE:
                boost = min(0.15, (s.tau_escalate - s.p_ema + 0.02) * 0.5)
                weights[lvl] = weights.get(lvl, 0) + boost
                total_boost += boost

        # BUG 5 FIX: also count current level mode
        s = self.states[current_level]
        mode = self.get_mode(current_level, s.p_ema)
        self.mode_counts[current_level][mode] += 1

        if total_boost > 0 and current_level in weights:
            weights[current_level] = max(0.2, weights[current_level] - total_boost)
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_threshold_snapshot(self) -> dict:
        return {
            lvl: {
                'tau_execute': round(s.tau_execute, 4),
                'tau_escalate': round(s.tau_escalate, 4),
                'p_ema': round(s.p_ema, 4),
            }
            for lvl, s in self.states.items()
        }

    def get_mode_counts(self) -> dict:
        return {lvl: dict(counts) for lvl, counts in self.mode_counts.items()}
