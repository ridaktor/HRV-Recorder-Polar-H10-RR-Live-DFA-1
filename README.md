<<<<<<< HEAD
# HRV-Recorder-Polar-H10-RR-Live-DFA-1
=======
# HRV Recorder (Polar H10) — RR + Live DFA α1

Small Python tool to log RR intervals from a Polar H10 over BLE and (optionally) compute **DFA α1** live, FatMaxxer-style.  
Includes live plots for **HR**, **RR**, and **DFA α1** with color-coded zones and optional **adaptive window** sizing.

https://github.com/yourname/hrv-recorder (replace with your repo)

---

## Features

- 🎧 Connects to **Polar H10** via BLE (macOS, Windows, Linux with Bleak).
- 🫀 Logs **RR intervals** (ms) with timestamps + **HR** snapshots to CSV.
- 📈 Live plots:
  - HR (bpm) — full session view, top-left live readout.
  - RR (ms) — rolling view, top-left live readout.
  - DFA α1 — FatMaxxer-style (scales 4–16 beats), background zones:
    - **Blue**: α1 > 0.75 (below LT1)
    - **Yellow**: 0.50–0.75 (LT1–LT2)
    - **Red**: α1 < 0.50 (≥ LT2)
- 🧠 **Adaptive α1 window** (optional): targets a fixed number of beats and adjusts window seconds based on current HR.
- 💾 CSV outputs default to `./data/` (auto-created).

---

## Installation

```bash
# Python 3.9+ recommended
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Start logging until Ctrl+C, compute α1 every 20 s with a 120 s window, show plots
python HRV.py --plot --alpha

--minutes M                 Duration in minutes (0 = until Ctrl+C). Default: 0
--out PATH                  RR CSV output (default: data/HRV_<timestamp>.csv)
--alpha_out PATH            α1 CSV output (default: data/HRV_<timestamp>_alpha.csv)
--device NAME               Substring of device name (e.g., "Polar" or "H10")
--address ADDR              Direct address/UUID (macOS often a long UUID)

--plot                      Show live plots (HR, RR, α1 when enabled)

--alpha                     Compute & plot DFA α1 live (FatMaxxer-style)
--alpha_window_s S          Fixed α1 window seconds (default: 120)
--alpha_step_s S            Recompute α1 every S seconds (default: 20)
--alpha_min_beats N         Minimum beats required in the window (default: 60)
--artifact_mode MODE        "auto" | "5" | "15" | "25" (% RR jump threshold)

--alpha_adaptive            Enable adaptive α1 window (targets beats)
--alpha_target_beats N      Target beats for adaptive window (default: 100)
--alpha_window_min_s S      Min seconds for adaptive window (default: 45)
--alpha_window_max_s S      Max seconds for adaptive window (default: 180)

Examples
1) Log with live α1 and plots (default windows):
python HRV.py --plot --alpha

2) Same but adaptive α1 window (aim for ~100 beats):
python HRV.py --plot --alpha --alpha_adaptive

3) Adaptive with tighter limits (60–150 s):
python HRV.py --plot --alpha --alpha_adaptive --alpha_target_beats 100 --alpha_window_min_s 60 --alpha_window_max_s 150

4) Run for 20 minutes only:
python HRV.py --plot --alpha --minutes 20

5) Connect by known address/UUID (macOS):
python HRV.py --plot --alpha --address DE4C7271-49D4-638E-173B-4A65AA13CF89

6) Save to custom paths:
python HRV.py --plot --alpha --out data/session_rr.csv --alpha_out data/session_alpha.csv
>>>>>>> 397e19f (Initial commit: Polar H10 RR logger with live DFA α1 (FatMaxxer-style) + adaptive window + plots)
