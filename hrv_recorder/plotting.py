import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter
from typing import Deque, Optional, Iterable


def _mmss(x, _pos):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m}:{s:02d}"


class LivePlotter:
    """
    Live HR/RR/DFA α1 plotter.

    - HR & RR use a rolling window of `window_seconds` (default 600s = 10 min).
    - DFA α1:
        * if alpha_full_history=True  -> full-session, accumulative plot (grows only)
        * if alpha_full_history=False -> same rolling window as HR/RR
    """

    def __init__(self, show_alpha: bool, window_seconds: int = 600, alpha_full_history: bool = True):
        self.win_s = float(window_seconds)
        self.alpha_full_history = bool(alpha_full_history)

        # Choose a GUI backend (before importing pyplot)
        try:
            matplotlib.use("MacOSX")
        except Exception:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                # fall back to default/headless if needed
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

        # Session clock
        self.clock_txt = self.fig.suptitle("Session 0:00", y=0.968)

        # ---------------- HR ----------------
        (self.hr_line,) = self.ax1.plot([], [], lw=1.5, color="red")
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
        self.ax2.set_xlim(0, self.win_s)
        self.rr_txt = self.ax2.text(
            0.01, 0.95, "", transform=self.ax2.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
        )

        # ---------------- α1 ----------------
        if show_alpha:
            # Zone backgrounds (blue / yellow / red)
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

            # Start with reasonable x-limits; for full history we will expand grow-only
            if not self.alpha_full_history:
                self.ax3.set_xlim(0, self.win_s)
            else:
                self.ax3.set_xlim(0.0, 60.0)

            self.a_txt = self.ax3.text(
                0.01, 0.95, "", transform=self.ax3.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
            )

        # --- NEW: persistent α1 buffers (accumulative) ---
        self._alpha_t = np.array([], dtype=float)       # absolute seconds (or monotonic timebase you pass)
        self._alpha_y = np.array([], dtype=float)
        self._alpha_base_t: Optional[float] = None      # first timestamp to anchor x=0

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
        if t is None or t.size != y.size:
            return
        x, y_plot = self._slice_last_window(np.asarray(t, float), y)
        if x.size == 0:
            return
        self.hr_line.set_data(x, y_plot)
        self.ax1.set_xlim(0, self.win_s)
        y_min, y_max = float(np.nanmin(y_plot)), float(np.nanmax(y_plot))
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
        """
        Accumulative α1:
        - Keeps a growing in-class history (self._alpha_t/_alpha_y).
        - When alpha_full_history=True, plots full session and grows xlim only.
        - When alpha_full_history=False, plots rolling window like HR/RR.
        """
        if self.ax3 is None:
            return

        # Convert incoming chunk
        t_in = np.asarray(list(times), float)
        y_in = np.asarray(list(values), float)
        if t_in.size == 0 or y_in.size == 0:
            return

        # Keep only finite points
        m = np.isfinite(t_in) & np.isfinite(y_in)
        if not np.any(m):
            return
        t_in, y_in = t_in[m], y_in[m]

        # Initialize base (anchor x=0 at the first α1 timestamp ever seen)
        if self._alpha_base_t is None:
            self._alpha_base_t = float(t_in[0])

        # ACCUMULATE: append, then sort & deduplicate by time (keep last for duplicates)
        if self._alpha_t.size == 0:
            self._alpha_t = t_in.copy()
            self._alpha_y = y_in.copy()
        else:
            self._alpha_t = np.concatenate([self._alpha_t, t_in])
            self._alpha_y = np.concatenate([self._alpha_y, y_in])
            order = np.argsort(self._alpha_t, kind="mergesort")
            self._alpha_t = self._alpha_t[order]
            self._alpha_y = self._alpha_y[order]
            if self._alpha_t.size > 1:
                keep = np.ones_like(self._alpha_t, dtype=bool)
                keep[:-1] = self._alpha_t[1:] != self._alpha_t[:-1]
                self._alpha_t = self._alpha_t[keep]
                self._alpha_y = self._alpha_y[keep]

        # Build x (elapsed seconds from first α1 point)
        x_all = self._alpha_t - self._alpha_base_t
        y_all = self._alpha_y

        if self.alpha_full_history:
            # Full session: never slice; extend xlim monotonically
            self.a_line.set_data(x_all, y_all)
            xmax_now = max(60.0, float(x_all[-1]))
            xmin_prev, xmax_prev = self.ax3.get_xlim()
            self.ax3.set_xlim(min(0.0, xmin_prev), max(xmax_prev, xmax_now))
            x_plot, y_plot = x_all, y_all
        else:
            # Rolling window for α1
            x_plot, y_plot = self._slice_last_window(x_all, y_all)
            if x_plot.size == 0:
                return
            self.a_line.set_data(x_plot, y_plot)
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


# ----------------------------------------------------------------
# Example usage (pseudo-code; replace with your data update loop):
# ----------------------------------------------------------------
# plotter = LivePlotter(show_alpha=True, window_seconds=600, alpha_full_history=True)
# while streaming:
#     # times are seconds (monotonic), arrays must line up with values
#     plotter.update_session_clock(elapsed_seconds)
#     plotter.update_hr(hr_array, t_hr_array)
#     plotter.update_rr(rr_deque, t_rr_deque)
#     plotter.update_alpha(alpha_times_array, alpha_values_array)  # <- accumulative!
#     plotter.draw()