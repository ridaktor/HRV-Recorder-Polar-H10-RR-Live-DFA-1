# hrv_recorder/dfa.py
from typing import Tuple
import numpy as np

def artifact_threshold_auto(hr_mean_bpm: float) -> float:
    """FatMaxxer-like jump threshold: >90 bpm -> 5%; <85 bpm -> 25%; else 15%."""
    if hr_mean_bpm > 90.0:
        return 0.05
    if hr_mean_bpm < 85.0:
        return 0.25
    return 0.15

def pick_threshold(mode: str, hr_mean_bpm: float) -> float:
    if mode == "auto":
        return artifact_threshold_auto(hr_mean_bpm)
    if mode == "5":
        return 0.05
    if mode == "15":
        return 0.15
    if mode == "25":
        return 0.25
    return artifact_threshold_auto(hr_mean_bpm)

def drop_artifacts(rr_ms: np.ndarray, thr: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Clamp RR to [300,2200] and drop |ΔRR|/prevRR > thr (relative jump).
    Returns (filtered_rr, artifact_fraction, keep_mask_for_original_window).
    """
    rr = np.asarray(rr_ms, float)
    n = rr.size
    if n == 0:
        return rr, 0.0, np.zeros(0, dtype=bool)

    valid = (rr > 300) & (rr < 2200)
    keep = valid.copy()
    rr_v = rr[valid]
    if rr_v.size >= 2:
        jumps = np.abs(np.diff(rr_v)) / np.maximum(rr_v[:-1], 1e-9)
        keep_valid = np.concatenate([[True], jumps <= thr])
        idx_valid = np.where(valid)[0]
        keep[idx_valid] &= keep_valid

    rr_f = rr[keep]
    art_frac = 1.0 - (rr_f.size / max(n, 1))
    return rr_f, float(art_frac), keep

def dfa_alpha1_short(rr_ms_window: np.ndarray) -> float:
    """
    DFA α1 on short-term scales 4..16 beats in the beat domain (FatMaxxer-style).
    """
    rr = np.asarray(rr_ms_window, float)
    rr = rr[(rr > 300) & (rr < 2200)]
    if rr.size < 50:
        return np.nan

    x = rr - rr.mean()
    y = np.cumsum(x)
    scales = np.arange(4, 17)  # beats
    Fs, Ns = [], []
    for n in scales:
        m = len(y) // n
        if m < 2:
            continue
        seg = y[:m*n].reshape(m, n)
        t = np.arange(n)
        t_mean = (n - 1) / 2.0
        y_mean = seg.mean(axis=1, keepdims=True)
        num = np.sum((t - t_mean) * (seg - y_mean), axis=1)
        den = np.sum((t - t_mean) ** 2)
        b = num / den
        a = (y_mean.flatten() - b * t_mean)
        detr = seg - (a[:, None] + b[:, None] * t[None, :])
        Fs.append(np.sqrt(np.mean(detr**2)))
        Ns.append(n)
    if not Fs:
        return np.nan
    Fs = np.asarray(Fs, float); Ns = np.asarray(Ns, float)
    good = np.isfinite(Fs) & (Fs > 0)
    if good.sum() < 3:
        return np.nan
    lx = np.log10(Ns[good]); ly = np.log10(Fs[good])
    return float(np.polyfit(lx, ly, 1)[0])