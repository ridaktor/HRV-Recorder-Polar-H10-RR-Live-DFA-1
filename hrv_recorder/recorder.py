import argparse
import asyncio
import csv
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Deque, Optional, Union

import numpy as np
from bleak import BleakClient
from bleak.exc import BleakDeviceNotFoundError

from .ble import HR_CHAR_UUID, parse_rr_intervals, find_device
from .dfa import (
    pick_threshold,
    drop_artifacts,
    dfa_alpha1_short,
    choose_adaptive_window_s,
)
from .plotting import LivePlotter


# ---------- CLI ----------
def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--minutes", type=float, default=0, help="Duration (0=until Ctrl+C)")
    ap.add_argument("--out", default="", help="RR CSV (default: data/HRV_<timestamp>.csv)")
    ap.add_argument("--alpha_out", default="", help="α1 CSV (default: data/HRV_<timestamp>_alpha.csv)")
    ap.add_argument("--device", default="", help="Substring of device name (e.g. 'Polar')")
    ap.add_argument("--address", default="", help="Direct address/UUID (macOS ok)")
    ap.add_argument("--plot", action="store_true", help="Show live plots")

    # Preset modes
    ap.add_argument("--mode", choices=["conservative", "quick", "adaptive"],
                    help="Preset configuration: conservative | quick | adaptive")

    # Time-based α1 (FatMaxxer classic)
    ap.add_argument("--alpha", action="store_true", help="Compute & plot DFA α1 (4..16 beats)")
    ap.add_argument("--alpha_window_s", type=int, default=120, help="α1 window seconds [FMX=120]")
    ap.add_argument("--alpha_step_s",   type=float, default=20.0,  help="Recompute α1 every N seconds [FMX=20]")
    ap.add_argument("--alpha_min_beats", type=int, default=60, help="Minimum beats to compute α1")
    ap.add_argument("--artifact_mode", choices=["auto","5","15","25"], default="auto",
                    help="RR jump threshold: auto=5/15/25%% by HR, or fixed (5/15/25)")
    ap.add_argument("--alpha_ramp_gate", type=float, default=0.0,
                    help="Skip α1 if |dHR/dt| exceeds this (bpm/min). Use 0 to disable.")
    ap.add_argument("--alpha_artifact_max_pct", type=float, default=1.0,
                    help="Skip α1 if artifact fraction > this (0..1). Use 1.0 to disable.")

    # Adaptive time-window (optional)
    ap.add_argument("--alpha_adaptive", action="store_true",
                    help="Adaptive α1 time window targeting beats")
    ap.add_argument("--alpha_target_beats", type=int, default=60,
                    help="Target beats when adaptive is enabled")
    ap.add_argument("--alpha_window_min_s", type=int, default=18,
                    help="Min seconds for adaptive window")
    ap.add_argument("--alpha_window_max_s", type=int, default=60,
                    help="Max seconds for adaptive window")

    # Beat-synchronous α1 (very fast)
    ap.add_argument("--alpha_follow_rr", action="store_true",
                    help="Recompute α1 on a sliding beat window")
    ap.add_argument("--alpha_window_beats", type=int, default=55,
                    help="Beat-count window for per-beat α1")
    ap.add_argument("--alpha_step_beats", type=int, default=1,
                    help="Recompute α1 every N beats")
    ap.add_argument("--alpha_smooth_pts", type=int, default=3,
                    help="Moving-average points for plotted α1 (plot only)")
    return ap


def _apply_mode_presets(args, defaults):
    """Set mode defaults only where the user kept the parser defaults."""
    if not getattr(args, "mode", None):
        return

    presets = {
        "conservative": {
            "alpha": True,
            "alpha_follow_rr": False,
            "alpha_window_s": 120,
            "alpha_step_s": 20.0,
            "alpha_min_beats": 60,
            "artifact_mode": "auto",
            "alpha_ramp_gate": 15.0,
            "alpha_artifact_max_pct": 0.10,
        },
        "quick": {
            "alpha": True,
            "alpha_follow_rr": True,
            "alpha_window_beats": 60,
            "alpha_step_beats": 1,
            "alpha_smooth_pts": 3,
            "alpha_min_beats": 50,
            "artifact_mode": "5",
        },
        "adaptive": {
            "alpha": True,
            "alpha_follow_rr": False,
            "alpha_adaptive": True,
            "alpha_target_beats": 60,
            "alpha_window_min_s": 18,
            "alpha_window_max_s": 60,
            "alpha_step_s": 2.0,
            "alpha_min_beats": 50,
            "artifact_mode": "15",
        },
    }

    cfg = presets[args.mode]
    for k, v in cfg.items():
        if k in defaults and getattr(args, k, None) == defaults[k]:
            setattr(args, k, v)


def _print_effective_config(args):
    print("Mode:", getattr(args, "mode", None))
    show = [
        "alpha","alpha_follow_rr","alpha_window_s","alpha_step_s","alpha_min_beats",
        "alpha_adaptive","alpha_target_beats","alpha_window_min_s","alpha_window_max_s",
        "alpha_window_beats","alpha_step_beats","alpha_smooth_pts",
        "artifact_mode","alpha_ramp_gate","alpha_artifact_max_pct",
    ]
    for k in show:
        if hasattr(args, k):
            print(f"  {k} = {getattr(args, k)}")


# ---------- Runner ----------
async def run():
    ap = add_args(argparse.ArgumentParser(description="Polar H10 RR logger + DFA α1 (FatMaxxer-style)"))

    # capture parser defaults before user overrides (for mode preset logic)
    parser_defaults = vars(ap.parse_args([]))
    args = ap.parse_args()
    _apply_mode_presets(args, parser_defaults)
    _print_effective_config(args)

    # --- output paths (default -> ./data)
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out.strip()) if args.out.strip() else (data_dir / f"HRV_{ts_tag}.csv")
    alpha_path = (Path(args.alpha_out.strip()) if args.alpha_out.strip()
                  else (data_dir / f"HRV_{ts_tag}_alpha.csv")) if args.alpha else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if alpha_path:
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging RR to: {out_path}")
    if alpha_path:
        print(f"Logging α1 to: {alpha_path}")

    # --- resolve target
    if args.address.strip():
        target: Union[str, object] = args.address.strip()
    else:
        print("Searching for Polar H10…")
        dev = await find_device(args.device if args.device else "polar")
        if dev is None:
            print("No device found. Wear/wet strap and make sure it is not connected elsewhere.")
            return
        target = dev

    # --- CSVs
    f_rr = out_path.open("w", newline="")
    w_rr = csv.writer(f_rr)
    w_rr.writerow(["timestamp_utc", "unix_ms", "rr_ms", "hr_bpm"])

    if alpha_path:
        f_a = alpha_path.open("w", newline="")
        w_a = csv.writer(f_a)
        w_a.writerow([
            "timestamp_utc", "unix_ms", "alpha1", "beats_used",
            "artifact_threshold", "artifact_fraction", "window_s_or_beats"
        ])
    else:
        f_a = None; w_a = None

    # --- buffers
    rr_buf: Deque[float] = deque(maxlen=20000)  # per-beat RR (ms)
    t_buf: Deque[float] = deque(maxlen=20000)   # per-beat POSIX time (s)
    hr_disp_buf: Deque[float] = deque(maxlen=20000)
    hr_t_buf: Deque[float] = deque(maxlen=20000)

    # α1 traces
    alpha_times: Deque[float] = deque(maxlen=10000)     # seconds since start
    alpha_vals:  Deque[float] = deque(maxlen=10000)     # used by time-based mode OR smoothed quick mode
    alpha_vals_raw:  Deque[float] = deque(maxlen=10000) # raw quick mode (for CSV + optional diagnostics)

    beats_saved = 0
    last_beat_time = [None]      # mutable for closure
    beats_since_alpha = [0]      # quick mode counter
    last_alpha_compute_s = [None]  # time-based mode

    # --- plotting (10-min rolling window)
    # --- plotting (10-min rolling window for HR/RR; full-session DFA)
    plotter = LivePlotter(
        show_alpha=bool(args.alpha),
        window_seconds=600,
        alpha_full_history=True  # <-- full-session DFA
    ) if args.plot else None
    t0 = datetime.now(timezone.utc).timestamp()

    def _smooth_for_plot(y: Deque[float], k: int) -> float:
        """Return last value or small moving average (plot only)."""
        if k <= 1 or len(y) < 2:
            return y[-1]
        n = min(k, len(y))
        arr = np.array(list(y)[-n:], float)
        return float(np.nanmean(arr))

    # --- BLE callback
    def callback(_: int, data: bytearray):
        nonlocal beats_saved
        rr_list, hr = parse_rr_intervals(bytes(data))
        now = datetime.now(timezone.utc)

        for rr_ms in rr_list:
            # build per-beat timestamp by accumulating RR
            if last_beat_time[0] is None:
                bt = now
            else:
                bt = last_beat_time[0] + timedelta(milliseconds=float(rr_ms))
            last_beat_time[0] = bt

            # write RR row
            ts_iso = bt.isoformat()
            unix_ms = int(bt.timestamp() * 1000)
            w_rr.writerow([ts_iso, unix_ms, rr_ms, hr])
            beats_saved += 1

            # update RR buffers
            rr_buf.append(float(rr_ms))
            t_buf.append(bt.timestamp())

            # update HR series (median of last 5 RR), timestamped at this beat
            last_rr5 = np.array(list(rr_buf)[-5:], float)
            if last_rr5.size > 0 and np.all(np.isfinite(last_rr5)):
                hr_est = 60000.0 / float(np.median(last_rr5))
                hr_disp_buf.append(hr_est)
                hr_t_buf.append(bt.timestamp())

            # ---- QUICK α1 (beat-synchronous) ----
            if args.alpha and args.alpha_follow_rr:
                beats_since_alpha[0] += 1
                if beats_since_alpha[0] >= max(1, int(args.alpha_step_beats)):
                    beats_since_alpha[0] = 0

                    # Take a tail slightly longer than window to allow artifact dropping
                    tail_len = max(args.alpha_window_beats * 2, args.alpha_min_beats)
                    rr_tail = np.array(list(rr_buf)[-tail_len:], float)

                    if rr_tail.size >= args.alpha_min_beats:
                        # Threshold from current mean HR
                        mean_rr = float(np.nanmean(rr_tail))
                        hr_mean = 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else 0.0
                        thr = pick_threshold(args.artifact_mode, hr_mean)

                        rr_clean, art_frac, _ = drop_artifacts(rr_tail, thr)

                        # Use last K cleaned beats
                        K = int(args.alpha_window_beats)
                        if rr_clean.size >= max(args.alpha_min_beats, K):
                            rr_win = rr_clean[-K:]

                            a1 = dfa_alpha1_short(rr_win)
                            if np.isfinite(a1):
                                t_rel = bt.timestamp() - t0
                                alpha_times.append(t_rel)
                                alpha_vals_raw.append(a1)

                                # smooth for plotting only
                                a1_plot = _smooth_for_plot(alpha_vals_raw, int(args.alpha_smooth_pts))
                                alpha_vals.append(a1_plot)

                                if w_a:
                                    w_a.writerow([
                                        ts_iso,
                                        unix_ms,
                                        f"{a1:.4f}",
                                        int(rr_win.size),
                                        f"{int(thr*100)}%",
                                        f"{art_frac:.3f}",
                                        f"{K}b",
                                    ])

    # --- connect + loop
    client = BleakClient(target)
    try:
        try:
            print("Connecting…")
            await client.connect()
        except BleakDeviceNotFoundError:
            print("Direct address failed; retrying discovery…")
            dev = await find_device(args.device if args.device else "polar")
            if dev is None:
                print("Rescan failed. Strap must be worn/wet, not paired elsewhere.")
                return
            client = BleakClient(dev)
            print("Connecting…")
            await client.connect()

        if not client.is_connected:
            print("Failed to connect.")
            return

        print("Connected. Starting notifications…")
        await client.start_notify(HR_CHAR_UUID, callback)

        elapsed = 0
        stop_after = int(args.minutes * 60) if args.minutes > 0 else None

        while True:
            await asyncio.sleep(1)
            elapsed += 1
            f_rr.flush()
            if f_a: f_a.flush()

            # ---- TIME-BASED α1 (classic/adaptive) ----
            if args.alpha and not args.alpha_follow_rr:
                now_s = datetime.now(timezone.utc).timestamp()
                if (last_alpha_compute_s[0] is None) or (now_s - last_alpha_compute_s[0] >= float(args.alpha_step_s)):
                    last_alpha_compute_s[0] = now_s

                    if len(t_buf) > 0:
                        # choose effective window seconds
                        if args.alpha_adaptive:
                            rr_tail = np.array(list(rr_buf)[-max(args.alpha_min_beats, 60):], float)
                            eff_window_s_opt = choose_adaptive_window_s(
                                rr_tail_ms=rr_tail,
                                target_beats=args.alpha_target_beats,
                                min_s=args.alpha_window_min_s,
                                max_s=args.alpha_window_max_s,
                            )
                            eff_window_s = float(eff_window_s_opt or args.alpha_window_s)
                        else:
                            eff_window_s = float(args.alpha_window_s)

                        # window selection
                        t_last = float(t_buf[-1])
                        t_start = t_last - eff_window_s
                        tt = np.array(list(t_buf), float)
                        rr_all = np.array(list(rr_buf), float)
                        mask = tt >= t_start
                        if np.any(mask):
                            rr_win = rr_all[mask]
                            t_win = tt[mask]  # for ramp gate
                            if rr_win.size >= args.alpha_min_beats:
                                # artifact threshold from mean HR
                                mean_rr = float(np.nanmean(rr_win))
                                hr_mean = 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else 0.0
                                thr = pick_threshold(args.artifact_mode, hr_mean)

                                rr_clean, art_frac, _ = drop_artifacts(rr_win, thr)

                                # ramp gate (bpm/min)
                                hr_win = 60000.0 / np.maximum(rr_win, 1e-6)
                                x = t_win - t_win.mean()
                                y = hr_win - hr_win.mean()
                                denom = float((x * x).sum())
                                hr_slope_bpm_per_min = 60.0 * float((x * y).sum() / denom) if denom > 0 else 0.0

                                ramp_bad = (args.alpha_ramp_gate > 0.0) and (abs(hr_slope_bpm_per_min) > args.alpha_ramp_gate)
                                art_bad  = (args.alpha_artifact_max_pct < 1.0) and (art_frac > args.alpha_artifact_max_pct)

                                if (not ramp_bad) and (not art_bad) and (rr_clean.size >= args.alpha_min_beats):
                                    a1 = dfa_alpha1_short(rr_clean)
                                    if np.isfinite(a1):
                                        alpha_times.append(now_s - t0)
                                        alpha_vals.append(a1)
                                        if w_a:
                                            w_a.writerow([
                                                datetime.now(timezone.utc).isoformat(),
                                                int(now_s * 1000),
                                                f"{a1:.4f}",
                                                int(rr_clean.size),
                                                f"{int(thr * 100)}%",
                                                f"{art_frac:.3f}",
                                                f"{float(eff_window_s):.1f}s",
                                            ])

            # ---- plots (10-min rolling window) ----
            if plotter:
                now_s = datetime.now(timezone.utc).timestamp()
                plotter.update_session_clock(elapsed_s=(now_s - t0))

                if len(hr_disp_buf) > 0:
                    y_hr = np.array(list(hr_disp_buf), float)
                    t_hr = np.array(list(hr_t_buf), float)
                    plotter.update_hr(y_hr, t_hr)

                plotter.update_rr(rr_buf, t_buf)

                if args.alpha:
                    plotter.update_alpha(alpha_times, alpha_vals)

                plotter.draw()

            if stop_after is not None and elapsed >= stop_after:
                break

            if elapsed % 10 == 0:
                # If in quick mode, report count of raw points; else count of time-based
                n_alpha = len(alpha_vals_raw) if args.alpha_follow_rr else len(alpha_vals)
                print(f"… {elapsed}s, beats saved: {beats_saved}, alpha points: {n_alpha}")

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)…")
    finally:
        try:
            await client.stop_notify(HR_CHAR_UUID)
        except Exception:
            pass
        try:
            await client.disconnect()
        except Exception:
            pass
        f_rr.flush(); f_rr.close()
        if f_a: f_a.flush(); f_a.close()
        print(f"Saved RR to {out_path}")
        if alpha_path:
            print(f"Saved α1 to {alpha_path}")
        print("Done.")