import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter
from typing import Deque, Optional, Iterable


def _mmss(x, _pos):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m}:{s:02d}"


class LivePlotter:
    def __init__(self, show_alpha: bool, window_seconds: int = 600, alpha_full_history: bool = True):
        """
        HR/RR use a rolling window (window_seconds).
        DFA α1 can show the full session (alpha_full_history=True) or the same rolling window.
        """
        self.win_s = float(window_seconds)
        self.alpha_full_history = bool(alpha_full_history)

        # Pick a GUI backend BEFORE importing pyplot
        try:
            matplotlib.use("MacOSX")
        except Exception:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass

        import matplotlib.pyplot as plt  # import after backend choice
        self.plt = plt
        self.plt.ion()

        if show_alpha:
            self.fig, (self.ax1, self.ax2, self.ax3) = self.plt.subplots(
                3, 1, figsize=(10, 8), sharex=False, constrained_layout=True
            )
        else:
            self.fig, (self.ax1, self.ax2) = self.plt.subplots(
                2, 1, figsize=(10, 7), sharex=False, constrained_layout=True
            )
            self.ax3 = None

        # Session clock (slightly lower so it’s more readable)
        self.clock_txt = self.fig.suptitle("Session 0:00", y=0.968)

        # ---------------- HR ----------------
        (self.hr_line,) = self.ax1.plot([], [], lw=1.5, color="red")  # red HR line
        self.ax1.set_ylabel("HR (bpm)")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, self.win_s)
        self.ax1.xaxis.set_major_formatter(FuncFormatter(_mmss))
        self.ax1.set_xlabel("Time (last 10 min, mm:ss)")
        self.hr_txt = self.ax1.text(
            0.01, 0.95, "", transform=self.ax1.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
        )

        # ---------------- RR ----------------
        (self.rr_line,) = self.ax2.plot([], [], lw=1.0)
        self.ax2.set_ylabel("RR (ms)")
        self.ax2.set_xlabel("Time (last 10 min, mm:ss)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.xaxis.set_major_formatter(FuncFormatter(_mmss))
        self.ax2.set_xlim(0, self.win_s)  # prevent negative ticks on startup
        self.rr_txt = self.ax2.text(
            0.01, 0.95, "", transform=self.ax2.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
        )

        # ---------------- α1 ----------------
        if show_alpha:
            # Zone backgrounds (Blue / Yellow / Red)
            self.ax3.axhspan(0.75, 2.0, facecolor="tab:blue", alpha=0.10, zorder=0)
            self.ax3.axhspan(0.50, 0.75, facecolor="yellow",  alpha=0.20, zorder=0)
            self.ax3.axhspan(0.00, 0.50, facecolor="red",     alpha=0.15, zorder=0)
            self.ax3.axhline(0.75, linestyle="--", linewidth=1, color="tab:blue", alpha=0.6)
            self.ax3.axhline(0.50, linestyle="--", linewidth=1, color="red",      alpha=0.6)

            (self.a_line,) = self.ax3.plot([], [], lw=1.8)
            self.ax3.set_ylim(0.2, 1.6)
            self.ax3.set_ylabel("DFA α1")
            self.ax3.set_xlabel("Time (mm:ss)")
            self.ax3.grid(True, alpha=0.3)
            self.ax3.xaxis.set_major_formatter(FuncFormatter(_mmss))

            # If NOT full history, start with rolling x-limits; otherwise, will expand dynamically.
            if not self.alpha_full_history:
                self.ax3.set_xlim(0, self.win_s)

            self.a_txt = self.ax3.text(
                0.01, 0.95, "", transform=self.ax3.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
            )

    # -------- helpers --------
    def _slice_last_window(self, t: np.ndarray, y: np.ndarray):
        """Return x in 0..win_s (oldest→now) and the matching y for last window."""
        if t.size == 0 or y.size == 0:
            return t, y
        t_last = t[-1]
        t0 = t_last - self.win_s
        mask = t >= t0
        if not np.any(mask):
            return np.array([]), np.array([])
        # Convert to positive window 0..win_s
        x_pos = t[mask] - t0
        return x_pos, y[mask]

    # -------- updates --------
    def update_session_clock(self, elapsed_s: float):
        m = int(elapsed_s // 60)
        s = int(elapsed_s % 60)
        self.clock_txt.set_text(f"Session {m}:{s:02d}")

    def update_hr(self, y: np.ndarray, t: Optional[np.ndarray] = None):
        if y.size == 0:
            return
        if t is not None and t.size == y.size:
            x, y_plot = self._slice_last_window(np.asarray(t, float), y)
            if x.size == 0:
                return
            self.hr_line.set_data(x, y_plot)
            self.ax1.set_xlim(0, self.win_s)
            y_min, y_max = float(np.nanmin(y_plot)), float(np.nanmax(y_plot))
        else:
            return
        pad = max(3.0, 0.05 * max(1.0, y_max - y_min))
        self.ax1.set_ylim(y_min - pad, y_max + pad)
        self.hr_txt.set_text(f"HR: {y[-1]:.0f} bpm")

    def update_rr(self, rr_buf: Deque[float], t_buf: Optional[Deque[float]] = None):
        if len(rr_buf) == 0 or t_buf is None or len(t_buf) != len(rr_buf):
            return
        t = np.array(list(t_buf), float)
        y = np.array(list(rr_buf), float)
        x, y_plot = self._slice_last_window(t, y)
        if x.size == 0:
            return
        self.rr_line.set_data(x, y_plot)
        self.ax2.set_xlim(0, self.win_s)
        y_min, y_max = float(np.nanmin(y_plot)), float(np.nanmax(y_plot))
        pad = max(10.0, 0.05 * max(1.0, y_max - y_min))
        self.ax2.set_ylim(y_min - pad, y_max + pad)
        self.rr_txt.set_text(f"RR: {y_plot[-1]:.0f} ms")

    def update_alpha(self, times: Iterable[float], values: Iterable[float]):
        if self.ax3 is None:
            return
        t = np.array(list(times), float)
        y = np.array(list(values), float)
        if t.size == 0 or y.size == 0:
            return

        if self.alpha_full_history:
            # Full-session DFA: no slicing; expand x-limits as we go
            x, y_plot = t, y
            self.a_line.set_data(x, y_plot)
            xmax = max(60.0, float(x[-1]))
            self.ax3.set_xlim(0.0, xmax)
        else:
            # Rolling 10-min DFA (legacy option)
            x, y_plot = self._slice_last_window(t, y)
            if x.size == 0:
                return
            self.a_line.set_data(x, y_plot)
            self.ax3.set_xlim(0, self.win_s)

        a_curr = y_plot[-1]
        self.a_txt.set_text(f"α1: {a_curr:.2f}")
        patch = self.a_txt.get_bbox_patch()
        if a_curr > 0.75:
            patch.set_facecolor("tab:blue");   patch.set_alpha(0.20)
        elif a_curr >= 0.50:
            patch.set_facecolor("yellow");     patch.set_alpha(0.25)
        else:
            patch.set_facecolor("red");        patch.set_alpha(0.20)

    def draw(self):
        try:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
        except Exception:
            pass