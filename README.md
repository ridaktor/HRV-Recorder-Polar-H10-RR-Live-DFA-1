# HRV Recorder (Polar H10) — RR + Live DFA α1

Small Python tool to log RR intervals from a Polar H10 over BLE and compute **DFA α1** live with three ready-to-use modes.  
Includes rolling 10-minute plots for **HR**, **RR**, and **DFA α1** with color-coded zones.

---

## What’s New (Defaults Tuned)

- DFA scale range now defaults to **4..12 beats** (closer match to FatMaxxer `alpha1v2`)
- New flag **`--dfa_nmax`** (default **12**) to tweak the upper scale
- Classic α1 timing preserved: **120 s** window, **20 s** step, **60** min beats, **artifact_mode=auto**

---

## Modes

- **conservative** — classic FatMaxxer style (120 s window, step 20 s). Smoother, robust; optional ramp/artifact gating.  
- **quick** — beat-synchronous α1 (e.g., 60-beat window, update every beat). Very low latency for intervals/Tabata.  
- **adaptive** — time window auto-sizes to ~60 beats (18–60 s), step 2 s. Faster response at high HR.

> You can override any preset with explicit flags.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or: pip install bleak numpy matplotlib
```

**BLE notes**
- **Windows/macOS:** enable Bluetooth; allow permissions on first run.  
- **Linux:** requires BlueZ; you may need extra caps:  
  `sudo setcap cap_net_raw+eip $(readlink -f $(which python))`

---

## Run

### Easiest (uses preset that enables α1)
```bash
python hrv.py --plot --mode conservative
```

### Or call the recorder directly
```bash
python recorder.py --alpha --plot
```

If running `recorder.py` directly, ensure it has an entrypoint at the bottom:

```python
if __name__ == "__main__":
    import asyncio
    from recorder import run
    asyncio.run(run())
```

---

## Defaults

- α1 (time-based): **window 120 s**, **step 20 s**, **min 60 beats**  
- Artifact handling: **`--artifact_mode auto`** (5/15/25% by HR)  
- DFA scales: **4..12** (tunable via `--dfa_nmax`)

---

## Manual Flags (advanced)

```bash
# Connection / IO
--minutes M                      # session length (0 = until Ctrl+C)
--out PATH                       # RR CSV (default: data/HRV_<timestamp>.csv)
--alpha_out PATH                 # α1 CSV (default: data/HRV_<timestamp>_alpha.csv)
--device NAME | --address ADDR   # discovery options
--plot                           # show live plots

# Time-based α1 (classic)
--alpha                          # enable α1
--alpha_window_s S               # default 120
--alpha_step_s S                 # default 20
--alpha_min_beats N              # default 60
--artifact_mode MODE             # auto | 5 | 15 | 25 (relative ΔRR threshold)
--alpha_ramp_gate BPM_PER_MIN    # skip windows with fast HR ramps (0 = off)
--alpha_artifact_max_pct PCT     # skip if artifacts > PCT (1.0 = off)

# Adaptive time window
--alpha_adaptive
--alpha_target_beats N           # default 60
--alpha_window_min_s S           # default 18
--alpha_window_max_s S           # default 60

# Beat-synchronous α1 (very fast)
--alpha_follow_rr
--alpha_window_beats K           # default 55
--alpha_step_beats N             # default 1
--alpha_smooth_pts K             # plotting-only moving average (default 3)

# DFA scale config (tuned default)
--dfa_nmax N                     # fit over 4..N beats (default 12)
```

---

## Outputs

- RR stream: `data/HRV_<timestamp>.csv`  
- α1 series: `data/HRV_<timestamp>_alpha.csv`

---

## Troubleshooting

- **Device not found:** make sure the strap is worn/wet and not paired elsewhere; try `--device polar` or `--address <MAC/UUID>`.  
- **No α1 early in session:** first α1 is emitted only after a full window (~120 s) with enough valid beats.
