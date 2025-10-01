# hrv_recorder/recorder.py
import argparse
import asyncio
import csv
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import numpy as np
from bleak import BleakClient
from bleak.exc import BleakDeviceNotFoundError

from .ble import HR_CHAR_UUID, parse_rr_intervals, find_device
from .dfa import pick_threshold, drop_artifacts, dfa_alpha1_short
from .plotting import LivePlotter

def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--minutes", type=float, default=0, help="Duration (0=until Ctrl+C)")
    ap.add_argument("--out", default="", help="RR CSV (default: data/HRV_<timestamp>.csv)")
    ap.add_argument("--alpha_out", default="", help="α1 CSV (default: data/HRV_<timestamp>_alpha.csv)")
    ap.add_argument("--device", default="", help="Substring of device name (e.g. 'Polar')")
    ap.add_argument("--address", default="", help="Direct address/UUID (macOS ok)")
    ap.add_argument("--plot", action="store_true", help="Show live plots")
    # FatMaxxer defaults:
    ap.add_argument("--alpha", action="store_true", help="Compute & plot DFA α1 (4..16 beats)")
    ap.add_argument("--alpha_window_s", type=int, default=120, help="α1 window seconds [FMX=120]")
    ap.add_argument("--alpha_step_s",   type=int, default=20,  help="Recompute α1 every N seconds [FMX=20]")
    ap.add_argument("--alpha_min_beats", type=int, default=60, help="Minimum beats to compute α1")
    ap.add_argument("--artifact_mode", choices=["auto","5","15","25"], default="auto",
                    help="RR jump threshold: auto=5/15/25%% by HR, or fixed (5/15/25)")
    return ap

async def run():
    ap = add_args(argparse.ArgumentParser(description="Polar H10 RR logger + DFA α1 (FatMaxxer-style)"))
    args = ap.parse_args()

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
        w_a.writerow(["timestamp_utc", "unix_ms", "alpha1", "beats_used",
                      "artifact_threshold", "artifact_fraction", "window_s"])
    else:
        f_a = None; w_a = None

    # --- buffers
    rr_buf: Deque[float] = deque(maxlen=20000)      # per-beat RR (ms)
    t_buf:  Deque[float] = deque(maxlen=20000)      # per-beat POSIX time (s)
    hr_disp_buf: Deque[float] = deque(maxlen=20000) # HR stream for plotting
    alpha_times: Deque[float] = deque(maxlen=1000)  # seconds since start
    alpha_vals:  Deque[float] = deque(maxlen=1000)
    beats_saved = 0
    last_beat_time = [None]  # mutable for closure

    # --- plotting
    plotter = LivePlotter(show_alpha=bool(args.alpha)) if args.plot else None
    t0 = datetime.now(timezone.utc).timestamp()

    # --- BLE callback
    def callback(_: int, data: bytearray):
        nonlocal beats_saved
        rr_list, hr = parse_rr_intervals(bytes(data))

        # A display HR: prefer median of last few RRs in this packet; else packet HR
        if rr_list:
            rr_for_hr = np.array(rr_list[-5:], float)
            if rr_for_hr.size > 0 and np.all(np.isfinite(rr_for_hr)):
                hr_disp_buf.append(60000.0 / float(np.median(rr_for_hr)))
        elif hr is not None:
            hr_disp_buf.append(float(hr))

        # Per-beat timestamps by accumulating RR from the last beat time
        for rr_ms in rr_list:
            if last_beat_time[0] is None:
                bt = datetime.now(timezone.utc)
            else:
                bt = last_beat_time[0] + timedelta(milliseconds=float(rr_ms))
            last_beat_time[0] = bt

            ts_iso = bt.isoformat()
            unix_ms = int(bt.timestamp() * 1000)

            w_rr.writerow([ts_iso, unix_ms, rr_ms, hr])
            beats_saved += 1

            rr_buf.append(float(rr_ms))
            t_buf.append(bt.timestamp())

    # --- connect + loop
    client = BleakClient(target)
    try:
        try:
            await client.connect()
        except BleakDeviceNotFoundError:
            print("Direct address failed; retrying discovery…")
            dev = await find_device(args.device if args.device else "polar")
            if dev is None:
                print("Rescan failed. Strap must be worn/wet, not paired elsewhere.")
                return
            client = BleakClient(dev)
            await client.connect()

        if not client.is_connected:
            print("Failed to connect."); return

        print("Connected. Starting notifications…")
        await client.start_notify(HR_CHAR_UUID, callback)

        elapsed = 0
        stop_after = int(args.minutes * 60) if args.minutes > 0 else None
        last_alpha_compute: Optional[float] = None

        while True:
            await asyncio.sleep(1)
            elapsed += 1
            f_rr.flush()
            if f_a: f_a.flush()

            # α1 every step
            if args.alpha:
                now_s = datetime.now(timezone.utc).timestamp()
                if (last_alpha_compute is None) or (now_s - last_alpha_compute >= args.alpha_step_s):
                    last_alpha_compute = now_s

                    if len(t_buf) > 0:
                        t_last = float(t_buf[-1])
                        t_start = t_last - float(args.alpha_window_s)
                        tt = np.array(list(t_buf), float)
                        rr_all = np.array(list(rr_buf), float)
                        mask = tt >= t_start
                        if np.any(mask):
                            rr_win = rr_all[mask]
                            if rr_win.size >= args.alpha_min_beats:
                                mean_rr = float(np.nanmean(rr_win))
                                hr_mean = 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else 0.0
                                thr = pick_threshold(args.artifact_mode, hr_mean)

                                rr_clean, art_frac, keep = drop_artifacts(rr_win, thr)

                                if rr_clean.size >= args.alpha_min_beats:
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
                                                f"{int(thr*100)}%",
                                                f"{art_frac:.3f}",
                                                f"{float(args.alpha_window_s):.1f}",
                                            ])

            # plots
            if plotter:
                if len(hr_disp_buf) > 0:
                    plotter.update_hr(np.fromiter(hr_disp_buf, dtype=float))
                plotter.update_rr(rr_buf)
                if args.alpha:
                    plotter.update_alpha(alpha_times, alpha_vals)
                plotter.draw()

            if stop_after is not None and elapsed >= stop_after:
                break

            if elapsed % 10 == 0:
                print(f"… {elapsed}s, beats saved: {beats_saved}, alpha points: {len(alpha_vals)}")

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