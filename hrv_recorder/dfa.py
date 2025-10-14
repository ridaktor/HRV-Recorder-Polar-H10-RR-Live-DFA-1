# dfa.py
from __future__ import annotations
import numpy as np
from typing import Tuple

# Default DFA scale range that best matches FatMaxxer (alpha1v2)
DFA_N_MIN_DEFAULT = 4
DFA_N_MAX_DEFAULT = 12


def compute_dfa_alpha1(
    rr_ms: np.ndarray,
    mode: str = "auto",
    hr_for_auto: float | None = None,
    n_min: int = DFA_N_MIN_DEFAULT,
    n_max: int = DFA_N_MAX_DEFAULT,
) -> tuple[float, float]:
    """
    Compute short-term DFA α1 on RR intervals.

    Returns:
        (alpha1, artifact_fraction)
    """
    rr = np.asarray(rr_ms, dtype=float)
    n = rr.size
    if n == 0:
        return (np.nan, 0.0)

    # ---- threshold (auto or fixed) ----
    if mode == "auto":
        if hr_for_auto is None:
            mean_rr = np.nanmean(rr)
            hr_mean_bpm = 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else 0.0
        else:
            hr_mean_bpm = hr_for_auto
        if hr_mean_bpm > 90.0:
            thr = 0.05
        elif hr_mean_bpm < 85.0:
            thr = 0.25
        else:
            thr = 0.15
    elif mode == "5":
        thr = 0.05
    elif mode == "15":
        thr = 0.15
    elif mode == "25":
        thr = 0.25
    else:
        # Fallback to auto mapping
        mean_rr = np.nanmean(rr)
        hr_mean_bpm = 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else 0.0
        thr = 0.05 if hr_mean_bpm > 90.0 else (0.25 if hr_mean_bpm < 85.0 else 0.15)

    # ---- artifact filter ----
    valid_mask = (rr > 300.0) & (rr < 2200.0)
    keep_mask = valid_mask.copy()
    rv = rr[valid_mask]
    if rv.size >= 2:
        jumps = np.abs(np.diff(rv)) / np.maximum(rv[:-1], 1e-9)
        keep_valid = np.concatenate([[True], jumps <= thr])
        idx_valid = np.where(valid_mask)[0]
        keep_mask[idx_valid] &= keep_valid
    rr_f = rr[keep_mask]
    art_frac = 1.0 - (rr_f.size / float(n))

    # ---- DFA α1 (short-term scales n_min..n_max beats) ----
    if rr_f.size < 50:
        return (np.nan, float(art_frac))

    # Integrated series
    y = np.cumsum(rr_f - rr_f.mean())

    F_vals, N_vals = [], []
    n_min = int(max(2, n_min))
    n_max = int(max(n_min + 1, n_max))
    for scale in range(n_min, n_max + 1):
        m = len(y) // scale
        if m < 2:
            continue
        seg = y[: m * scale].reshape(m, scale)
        t = np.arange(scale)
        t_mean = (scale - 1) / 2.0
        seg_mean = seg.mean(axis=1, keepdims=True)
        num = np.sum((t - t_mean) * (seg - seg_mean), axis=1)
        den = np.sum((t - t_mean) ** 2)
        b = num / den
        a = seg_mean.flatten() - b * t_mean
        detr = seg - (a[:, None] + b[:, None] * t[None, :])
        F = np.sqrt(np.mean(detr**2))
        if np.isfinite(F) and F > 0:
            F_vals.append(float(F))
            N_vals.append(int(scale))

    if len(F_vals) < 3:
        return (np.nan, float(art_frac))

    F_vals = np.asarray(F_vals, dtype=float)
    N_vals = np.asarray(N_vals, dtype=float)
    alpha1, _ = np.polyfit(np.log10(N_vals), np.log10(F_vals), 1)
    return (float(alpha1), float(art_frac))