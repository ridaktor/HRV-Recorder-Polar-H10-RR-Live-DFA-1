# HRV Recorder (Polar H10) — RR + Live DFA α1

Small Python tool to log RR intervals from a Polar H10 over BLE and compute **DFA α1** live with three ready-to-use modes.  
Includes rolling 10-minute plots for **HR**, **RR**, and **DFA α1** with color-coded zones.

---

## Modes

- **conservative** — classic FatMaxxer style (120 s window, step 20 s). Smoother, robust; optional ramp/artifact gating.
- **quick** — beat-synchronous α1 (e.g., 60-beat window, update every beat). Very low latency for intervals/Tabata.
- **adaptive** — time window auto-sizes to ~60 beats (18–60 s), step 2 s. Faster response at high HR.

You can override any preset with explicit flags.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Run
```bash
python hrv.py --plot --mode conservative
```

## Manual Flags (advanced)
```bash

# (Connection/IO)
--minutes M — session length (0 = until Ctrl+C)
--out PATH — RR CSV (default: data/HRV_<timestamp>.csv)
--alpha_out PATH — α1 CSV (default: data/HRV_<timestamp>_alpha.csv)
--device NAME / --address ADDR — device discovery options
--plot — show live plots

# Time-based α1 (classic)
--alpha — enable α1
--alpha_window_s S — window seconds (default 120)
--alpha_step_s S — recompute every S seconds (default 20)
--alpha_min_beats N — min beats within window (default 60)
--artifact_mode MODE — auto | 5 | 15 | 25 (relative ΔRR threshold)
--alpha_ramp_gate BPM_PER_MIN — skip windows with fast HR ramps (0=off)
--alpha_artifact_max_pct PCT — skip windows if artifacts > PCT (1.0=off)

# Adaptive time window
--alpha_adaptive — target beats instead of fixed seconds
--alpha_target_beats N — default 60
--alpha_window_min_s S / --alpha_window_max_s S — bounds (default 18–60 s)

# Beat-synchronous α1 (very fast)
--alpha_follow_rr — compute on a sliding beat window
--alpha_window_beats K — window size in beats (default 55)
--alpha_step_beats N — update every N beats (default 1)
--alpha_smooth_pts K — plotting-only moving average (default 3)