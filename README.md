# HRV Recorder (Polar H10) — RR + Live DFA α1

Simple Python tool to log **RR intervals** from a Polar H10 over BLE and compute short-term **DFA α1** live (FatMaxxer-style).

- Beat-accurate RR logging (one row per beat)
- DFA α1 on short-term scales **4–16 beats** (beat domain), window **120 s**, step **20 s**
- Simple artifact rule (auto **5/15/25%** RR-jump threshold based on mean HR)
- Minimal live plots for **HR**, **RR**, and **α1**
- CSV outputs in `./data/`

## Install

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt 
```

## Run

```bash
python hrv.py --plot --alpha

#Options:
--minutes M                 Duration (0 = until Ctrl+C). Default: 0
--out PATH                  RR CSV (default: data/HRV_<timestamp>.csv)
--alpha_out PATH            α1 CSV (default: data/HRV_<timestamp>_alpha.csv)
--device NAME               Substring of device name (e.g., "Polar")
--address ADDR              Direct address/UUID (macOS)

--plot                      Show live plots
--alpha                     Compute DFA α1 live (4–16 beats)
--alpha_window_s S          α1 window seconds (default: 120)
--alpha_step_s S            Recompute every S seconds (default: 20)
--alpha_min_beats N         Minimum beats to compute α1 (default: 60)
--artifact_mode MODE        "auto" | "5" | "15" | "25" (RR jump threshold)