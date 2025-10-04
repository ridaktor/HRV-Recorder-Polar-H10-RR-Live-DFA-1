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