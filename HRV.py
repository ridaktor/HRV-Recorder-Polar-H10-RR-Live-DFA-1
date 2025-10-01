# HRV.py
# pip install bleak numpy matplotlib

import asyncio
import argparse
import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Union, Deque
from collections import deque

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakDeviceNotFoundError

# ---- BLE UUIDs ----
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHAR_UUID    = "00002a37-0000-1000-8000-00805f9b34fb"


# ---------- Parse BLE Heart Rate Measurement (0x2A37) ----------
def parse_rr_intervals(data: bytes) -> Tuple[List[float], Optional[int]]:
    """Return (list of RR in ms, HR bpm or None) from Heart Rate Measurement packet."""
    if not data:
        return [], None
    flags = data[0]
    hr_16bit   = bool(flags & 0x01)
    rr_present = bool(flags & 0x10)

    idx = 1
    if hr_16bit:
        if len(data) < 3:
            return [], None
        hr = int.from_bytes(data[idx:idx+2], "little"); idx += 2
    else:
        if len(data) < 2:
            return [], None
        hr = data[idx]; idx += 1

    rr_list: List[float] = []
    if rr_present:
        while idx + 1 < len(data):
            rr_1_1024 = int.from_bytes(data[idx:idx+2], "little")  # 1/1024 s
            idx += 2
            rr_ms = rr_1_1024 * 1000.0 / 1024.0
            rr_list.append(rr_ms)

    return rr_list, int(hr) if hr is not None else None


# ---------- Robust discovery across Bleak/macOS variants ----------
async def _discover_normalized(timeout: float):
    """
    Return a list of (identifier, name, adv_service_uuids).
    identifier is a BLEDevice or an address/UUID string (macOS).
    """
    try:
        res = await BleakScanner.discover(timeout=timeout, return_adv=True)  # type: ignore[arg-type]
        items = []
        if isinstance(res, dict):  # CoreBluetooth may return a dict
            for k, v in res.items():
                dev = None; name = None; uuids = []
                if hasattr(k, "address"):  # BLEDevice as key
                    dev = k
                    name = (getattr(dev, "name", "") or "")
                    uuids = [u.lower() for u in (getattr(v, "service_uuids", None) or [])]
                    items.append((dev, name, uuids))
                elif isinstance(k, str):   # address/UUID as key
                    adv = v
                    if isinstance(adv, tuple) and len(adv) == 2 and hasattr(adv[0], "address"):
                        dev = adv[0]
                        name = (getattr(dev, "name", "") or "")
                        uuids = [u.lower() for u in (getattr(adv[1], "service_uuids", None) or [])]
                        items.append((dev, name, uuids))
                    else:
                        name = (getattr(adv, "local_name", "") or "")
                        uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
                        items.append((k, name, uuids))
        elif isinstance(res, (list, tuple)):
            items = []
            for it in res:
                if isinstance(it, tuple) and len(it) == 2 and hasattr(it[0], "address"):
                    dev, adv = it
                    name = (getattr(dev, "name", "") or "")
                    uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
                    items.append((dev, name, uuids))
                else:
                    dev = it
                    name = (getattr(dev, "name", "") or "")
                    items.append((dev, name, []))
        else:
            items = []
        return items
    except TypeError:
        devices = await BleakScanner.discover(timeout=timeout)
        items = []
        for d in devices:
            name = (getattr(d, "name", "") or "")
            items.append((d, name, []))
        return items


async def find_device(name_hint: Optional[str], timeout: float = 15.0) -> Optional[Union[str, "BLEDevice"]]:
    """
    Find Polar H10 by name or HR service UUID.
    Returns BLEDevice or address/UUID string (macOS).
    """
    print(f"Scanning for Polar H10… ({int(timeout)}s)")
    items = await _discover_normalized(timeout)
    nh = (name_hint or "").lower()
    cands = []
    for ident, name, uuids in items:
        nl = (name or "").lower()
        if nh and nh in nl:
            cands.append(ident); continue
        if "polar" in nl:
            cands.append(ident); continue
        if uuids and HR_SERVICE_UUID.lower() in uuids:
            cands.append(ident)

    if cands:
        pretty = []
        for c in cands:
            if hasattr(c, "address"):
                pretty.append((getattr(c, "name", None), getattr(c, "address", None)))
            else:
                pretty.append(("addr/uuid", str(c)))
        print("Found:", pretty)
        return cands[0]

    print("No suitable device found. Strap wet/snug and not connected elsewhere.")
    return None


# ---------- FatMaxxer-style DFA α1 ----------
def _artifact_threshold_auto(hr_mean_bpm: float) -> float:
    """Auto jump threshold: >90 bpm -> 5%; <85 bpm -> 25%; else 15%."""
    if hr_mean_bpm > 90.0:
        return 0.05
    if hr_mean_bpm < 85.0:
        return 0.25
    return 0.15

from typing import Tuple as TypingTuple
def _drop_artifacts_rr(rr_ms: np.ndarray, thr: float) -> TypingTuple[np.ndarray, float]:
    """Drop beats where |ΔRR|/prevRR > thr. Also clamp RR to [300, 2200] ms."""
    rr = np.asarray(rr_ms, float)
    rr = rr[(rr > 300) & (rr < 2200)]
    if rr.size < 3:
        return rr, 0.0
    keep = np.ones(rr.size, dtype=bool)
    jumps = np.abs(np.diff(rr)) / np.maximum(rr[:-1], 1e-9)
    keep[1:] &= (jumps <= thr)
    rr_f = rr[keep]
    art_frac = 1.0 - (rr_f.size / max(rr.size, 1))
    return rr_f, float(art_frac)

def dfa_alpha1_fmx(rr_ms_window: np.ndarray) -> float:
    """DFA α1 with short-term scales 4..16 beats (beat domain)."""
    scales = np.arange(4, 17)
    rr = np.asarray(rr_ms_window, float)
    if rr.size < 50:
        return np.nan
    rr = rr[(rr > 300) & (rr < 2200)]
    if rr.size < 50:
        return np.nan

    x = rr - rr.mean()
    y = np.cumsum(x)
    Fs, Ns = [], []
    for n in scales:
        if n < 2 or n > len(y):
            continue
        m = len(y) // n
        if m < 2:
            continue
        y_cut = y[:m*n].reshape(m, n)
        t = np.arange(n)
        t_mean = (n - 1) / 2.0
        y_mean = y_cut.mean(axis=1, keepdims=True)
        num = np.sum((t - t_mean) * (y_cut - y_mean), axis=1)
        den = np.sum((t - t_mean) ** 2)
        b = num / den
        a = (y_mean.flatten() - b * t_mean)
        fit = a[:, None] + b[:, None] * t[None, :]
        detr = y_cut - fit
        F_n = np.sqrt(np.mean(detr**2))
        Fs.append(F_n); Ns.append(n)
    if not Fs:
        return np.nan
    Fs = np.array(Fs, float); Ns = np.array(Ns, float)
    good = np.isfinite(Fs) & (Fs > 0)
    if good.sum() < 3:
        return np.nan
    lx = np.log10(Ns[good]); ly = np.log10(Fs[good])
    return float(np.polyfit(lx, ly, 1)[0])


# ---------- Adaptive window helper ----------
def _choose_adaptive_window_s(rr_tail_ms: np.ndarray,
                              target_beats: int,
                              min_s: int,
                              max_s: int) -> Optional[float]:
    """
    Choose a window (seconds) so that the window contains ~target_beats
    given the recent mean RR (ms). Clipped to [min_s, max_s].
    Returns None if insufficient data to estimate.
    """
    rr = np.asarray(rr_tail_ms, float)
    rr = rr[(rr > 300) & (rr < 2200)]
    if rr.size < 10:
        return None
    mean_rr_ms = float(np.nanmean(rr))
    if not np.isfinite(mean_rr_ms) or mean_rr_ms <= 0:
        return None
    window_s = (target_beats * mean_rr_ms) / 1000.0
    return float(np.clip(window_s, min_s, max_s))


# ---------- Main ----------
async def main():
    import matplotlib
    # Select a GUI backend BEFORE importing pyplot
    try:
        matplotlib.use("MacOSX")
    except Exception:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Log RR from Polar H10 to CSV (+ live HR/RR/α1, robust gating)")

    # Session + IO
    ap.add_argument("--minutes", type=float, default=0, help="Duration (0 = until Ctrl+C)")
    ap.add_argument("--out", default="", help="RR CSV (default: data/HRV_<timestamp>.csv)")
    ap.add_argument("--alpha_out", default="", help="Alpha CSV (default: data/HRV_<timestamp>_alpha.csv)")
    ap.add_argument("--device", default="", help="Substring of device name, e.g. 'Polar' or 'H10'")
    ap.add_argument("--address", default="", help="Direct device UUID/address (macOS)")
    ap.add_argument("--plot", action="store_true", help="Show live plots")

    # DFA options
    ap.add_argument("--alpha", action="store_true", help="Compute & plot DFA α1 live")
    ap.add_argument("--alpha_window_s", type=int, default=120, help="Fixed α1 window (seconds)")
    ap.add_argument("--alpha_step_s",   type=int, default=20,  help="Recompute α1 every N seconds")
    ap.add_argument("--alpha_min_beats", type=int, default=60, help="Minimum beats in window for α1")

    # Artifact + ramp gates
    ap.add_argument("--artifact_mode", choices=["auto","5","15","25"], default="auto",
                    help="Jump threshold: 5%%, 15%%, 25%% or auto (HR>90→5%%, HR<85→25%%, else 15%%)")
    ap.add_argument("--artifact_max_pct", type=float, default=0.10,
                    help="Skip α1 if > this fraction of beats are >20%% RR jumps (0.0–1.0)")
    ap.add_argument("--gate_hr_slope_bpm_min", type=float, default=2.0,
                    help="Skip α1 when |dHR/dt| in window exceeds this (bpm/min)")

    # Adaptive window
    ap.add_argument("--alpha_adaptive", action="store_true",
                    help="Enable adaptive α1 window (targets beats rather than seconds).")
    ap.add_argument("--alpha_target_beats", type=int, default=100,
                    help="Target beats for α1 when adaptive is enabled.")
    ap.add_argument("--alpha_window_min_s", type=int, default=45,
                    help="Minimum window seconds for adaptive α1.")
    ap.add_argument("--alpha_window_max_s", type=int, default=180,
                    help="Maximum window seconds for adaptive α1.")

    args = ap.parse_args()

    # --- Default output to ./data ---
    repo_dir = Path(__file__).resolve().parent
    data_dir = repo_dir / "data"
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

    # Resolve target
    if args.address.strip():
        target: Union[str, object] = args.address.strip()
    else:
        dev = await find_device(args.device if args.device else "Polar")
        if dev is None:
            sys.exit(1)
        target = dev

    # CSV writers
    f_rr = out_path.open("w", newline="")
    w_rr = csv.writer(f_rr)
    w_rr.writerow(["timestamp_utc", "unix_ms", "rr_ms", "hr_bpm"])
    f_rr.flush()

    if alpha_path:
        f_a = alpha_path.open("w", newline="")
        w_a = csv.writer(f_a)
        w_a.writerow(["timestamp_utc", "unix_ms", "alpha1", "beats_used",
                      "artifact_threshold", "artifact_fraction", "window_s"])
        f_a.flush()
    else:
        f_a = None; w_a = None

    # Buffers (beat-aligned)
    rr_buf: Deque[float] = deque(maxlen=20000)      # RR (ms)
    t_buf:  Deque[float] = deque(maxlen=20000)      # POSIX seconds for each beat
    hr_disp_buf: Deque[float] = deque(maxlen=20000) # HR display stream

    alpha_times: Deque[float] = deque(maxlen=1000)  # seconds since start
    alpha_vals:  Deque[float] = deque(maxlen=1000)

    beats_saved = 0
    last_beat_time = [None]  # mutable cell for per-beat time accumulation

    # BLE callback (write beats directly to CSV to avoid memory growth)
    def callback(_: int, data: bytearray):
        nonlocal beats_saved
        rr_list, hr = parse_rr_intervals(bytes(data))

        # Update display HR: prefer beat-derived median if we got beats; else packet HR
        if rr_list:
            rr_for_hr = np.array(rr_list[-5:], float)
            if rr_for_hr.size > 0 and np.all(np.isfinite(rr_for_hr)):
                hr_disp_buf.append(60000.0 / float(np.median(rr_for_hr)))
        elif hr is not None:
            hr_disp_buf.append(float(hr))

        # Beat-by-beat timestamps (accumulate RR)
        for rr_ms in rr_list:
            if last_beat_time[0] is None:
                bt = datetime.now(timezone.utc)
            else:
                bt = last_beat_time[0] + timedelta(milliseconds=float(rr_ms))
            last_beat_time[0] = bt

            ts_iso = bt.isoformat()
            unix_ms = int(bt.timestamp() * 1000)

            # Write immediately (memory-safe)
            w_rr.writerow([ts_iso, unix_ms, rr_ms, hr])
            beats_saved += 1

            # Update buffers for analysis/plots
            rr_buf.append(float(rr_ms))
            t_buf.append(bt.timestamp())

    # Plot setup
    if args.plot:
        plt.ion()
        if args.alpha:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
            ax3 = None

        # Lines
        hr_line, = ax1.plot([], [], lw=1.5, color="tab:blue")
        rr_line, = ax2.plot([], [], lw=1.0, color="tab:orange")

        # Readouts
        hr_txt = ax1.text(0.01, 0.95, "", transform=ax1.transAxes, va="top",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
        rr_txt = ax2.text(0.01, 0.95, "", transform=ax2.transAxes, va="top",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        # Labels/grids
        ax1.set_ylabel("HR (bpm)"); ax1.grid(True, alpha=0.3)
        ax2.set_ylabel("RR (ms)");  ax2.set_xlabel("Samples"); ax2.grid(True, alpha=0.3)

        if args.alpha:
            # Zones
            ax3.axhspan(0.75, 2.0, facecolor="blue",   alpha=0.08, zorder=0)
            ax3.axhspan(0.50, 0.75, facecolor="yellow", alpha=0.15, zorder=0)
            ax3.axhspan(0.00, 0.50, facecolor="red",    alpha=0.12, zorder=0)
            ax3.axhline(0.75, linestyle="--", linewidth=1, color="red")
            ax3.axhline(0.50, linestyle="--", linewidth=1, color="red")

            a_line, = ax3.plot([], [], lw=1.8, color="red", zorder=5)
            ax3.set_ylim(0.2, 1.6)
            ax3.set_ylabel("DFA α1"); ax3.set_xlabel("Time (s)"); ax3.grid(True, alpha=0.3)

            a_txt = ax3.text(0.01, 0.95, "", transform=ax3.transAxes, va="top",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    # Connect (with fallback if address became stale on macOS)
    client = BleakClient(target)
    try:
        try:
            await client.connect()
        except BleakDeviceNotFoundError:
            print("Direct address failed; rescanning by name…")
            dev = await find_device(args.device if args.device else "Polar")
            if dev is None:
                print("Rescan failed. Ensure strap is worn, wet, and not connected elsewhere.")
                return
            client = BleakClient(dev)
            await client.connect()

        if not client.is_connected:
            print("Failed to connect.")
            return
        print("Connected. Starting notifications…")
        await client.start_notify(HR_CHAR_UUID, callback)

        elapsed = 0
        stop_after = int(args.minutes * 60) if args.minutes > 0 else None
        last_alpha_compute = None
        t0 = datetime.now(timezone.utc).timestamp()

        while True:
            await asyncio.sleep(1)
            elapsed += 1

            # Keep RR file on disk up to date
            f_rr.flush()

            # α1 (fixed or adaptive) every alpha_step_s
            if args.alpha:
                now = datetime.now(timezone.utc)
                if (last_alpha_compute is None) or ((now - last_alpha_compute).total_seconds() >= args.alpha_step_s):
                    last_alpha_compute = now

                    # Need some data and minimum beats
                    if len(t_buf) > 0 and len(rr_buf) >= args.alpha_min_beats:

                        # --- choose effective window seconds (fixed or adaptive) ---
                        if getattr(args, "alpha_adaptive", False):
                            rr_tail_for_hr = np.array(list(rr_buf)[-max(args.alpha_min_beats, 60):], float)
                            eff_window_s = _choose_adaptive_window_s(
                                rr_tail_for_hr,
                                getattr(args, "alpha_target_beats", 100),
                                getattr(args, "alpha_window_min_s", 45),
                                getattr(args, "alpha_window_max_s", 180),
                            )
                            if eff_window_s is None:
                                eff_window_s = float(args.alpha_window_s)
                        else:
                            eff_window_s = float(args.alpha_window_s)

                        # --- build window mask using beat-aligned timestamps ---
                        t_last = float(t_buf[-1])  # POSIX seconds
                        t_start = t_last - eff_window_s
                        tt = np.array(list(t_buf), float)
                        rr_all = np.array(list(rr_buf), float)

                        mask = tt >= t_start
                        if not np.any(mask):
                            continue

                        rr_win = rr_all[mask]
                        t_win = tt[mask]

                        # still need enough beats pre-clean
                        if rr_win.size < args.alpha_min_beats:
                            continue

                        # --- derive instant HR from RR for slope estimate (bpm/min) ---
                        hr_win = 60000.0 / np.maximum(rr_win, 1e-6)
                        x = t_win - t_win.mean()
                        y = hr_win - hr_win.mean()
                        denom = float((x * x).sum())
                        hr_slope_bpm_per_min = 60.0 * float((x * y).sum() / denom) if denom > 0 else 0.0

                        # thresholds
                        gate_hr_slope_bpm_min = float(getattr(args, "gate_hr_slope_bpm_min", 2.0))
                        artifact_max_pct      = float(getattr(args, "artifact_max_pct", 0.10))

                        # --- artifact jump threshold selection (FatMaxxer-like) ---
                        mean_rr = float(np.nanmean(rr_win))
                        hr_mean = 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else 0.0
                        if args.artifact_mode == "auto":
                            thr = _artifact_threshold_auto(hr_mean)  # 5/15/25% depending on HR
                        elif args.artifact_mode == "5":
                            thr = 0.05
                        elif args.artifact_mode == "15":
                            thr = 0.15
                        else:  # "25"
                            thr = 0.25

                        # --- clean RR + compute artifact fraction in this window ---
                        rr_clean, art_frac = _drop_artifacts_rr(rr_win, thr)

                        # gating: steady-state + acceptable noise + enough beats AFTER cleaning
                        if (abs(hr_slope_bpm_per_min) > gate_hr_slope_bpm_min) or \
                           (art_frac > artifact_max_pct) or \
                           (rr_clean.size < args.alpha_min_beats):
                            # Optional debug:
                            # print(f"Skip α1: slope={hr_slope_bpm_per_min:.1f}, art={art_frac:.2f}, beats={rr_clean.size}")
                            pass
                        else:
                            # --- compute DFA α1 (short-term 4..16 beats) ---
                            a1 = dfa_alpha1_fmx(rr_clean)
                            if np.isfinite(a1):
                                alpha_times.append(now.timestamp() - t0)  # seconds since start
                                alpha_vals.append(a1)
                                if w_a:
                                    w_a.writerow([
                                        now.isoformat(),
                                        int(now.timestamp() * 1000),
                                        f"{a1:.4f}",
                                        int(rr_clean.size),
                                        f"{int(thr * 100)}%",
                                        f"{art_frac:.3f}",
                                        f"{eff_window_s:.1f}",
                                    ])
                                    f_a.flush()

            # Live plots
            if args.plot:
                # HR — plot ALL samples, set limits, show current value
                if len(hr_disp_buf) > 0:
                    y = np.fromiter(hr_disp_buf, dtype=float)
                    x_arr = np.arange(y.size)
                    hr_line.set_data(x_arr, y)

                    ax1.set_xlim(0, max(50, int(x_arr[-1])))
                    y_min = float(np.nanmin(y)); y_max = float(np.nanmax(y))
                    pad = max(3.0, 0.05 * max(1.0, y_max - y_min))
                    ax1.set_ylim(y_min - pad, y_max + pad)

                    hr_curr = y[-1]
                    hr_txt.set_text(f"HR: {hr_curr:.0f} bpm")

                # RR — rolling window for readability, show current value
                if len(rr_buf) > 0:
                    y2 = np.array(list(rr_buf)[-2000:], float)
                    x2 = np.arange(y2.size)
                    rr_line.set_data(x2, y2)

                    ax2.set_xlim(0, max(50, int(x2[-1])))
                    y2_min = float(np.nanmin(y2)); y2_max = float(np.nanmax(y2))
                    pad2 = max(10.0, 0.05 * max(1.0, y2_max - y2_min))
                    ax2.set_ylim(y2_min - pad2, y2_max + pad2)

                    rr_curr = y2[-1]
                    rr_txt.set_text(f"RR: {rr_curr:.0f} ms")

                # α1 timeline (seconds since start) + colored box
                if args.alpha and len(alpha_vals) > 0:
                    x3 = np.array(list(alpha_times), float)
                    y3 = np.array(list(alpha_vals), float)
                    a_line.set_data(x3, y3)

                    a_curr = y3[-1]
                    a_txt.set_text(f"α1: {a_curr:.2f}")

                    # set box background color by zone
                    patch = a_txt.get_bbox_patch()
                    if a_curr > 0.75:
                        patch.set_facecolor("blue");   patch.set_alpha(0.20)
                    elif a_curr >= 0.50:
                        patch.set_facecolor("yellow"); patch.set_alpha(0.25)
                    else:
                        patch.set_facecolor("red");    patch.set_alpha(0.20)

                    # keep fixed y-limits; autoscale only X
                    ax3.relim()
                    ax3.autoscale_view(scalex=True, scaley=False)

                try:
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                except Exception:
                    pass

            if stop_after is not None and elapsed >= stop_after:
                break

            if elapsed % 10 == 0:
                print(f"… {elapsed}s, beats saved: {beats_saved}, alpha points: {len(alpha_vals)}")

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C). Writing remaining data…")
    finally:
        try:
            await client.stop_notify(HR_CHAR_UUID)
        except Exception:
            pass
        try:
            await client.disconnect()
        except Exception:
            pass

        # Final flushes
        f_rr.flush(); f_rr.close()
        if alpha_path:
            f_a.flush(); f_a.close()
        print(f"Saved RR to {out_path}")
        if alpha_path:
            print(f"Saved α1 to {alpha_path}")
        print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass