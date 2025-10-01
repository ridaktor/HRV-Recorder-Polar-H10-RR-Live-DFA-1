# hrv_recorder/plotting.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LivePlotter:
    def __init__(self, show_alpha: bool):
        # Pick a GUI backend before pyplot (best-effort)
        try:
            matplotlib.use("MacOSX")
        except Exception:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass

        plt.ion()
        if show_alpha:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
        else:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
            self.ax3 = None

        # HR
        self.hr_line, = self.ax1.plot([], [], lw=1.5)
        self.ax1.set_ylabel("HR (bpm)"); self.ax1.grid(True, alpha=0.3)
        self.hr_txt = self.ax1.text(0.01, 0.95, "", transform=self.ax1.transAxes, va="top",
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        # RR
        self.rr_line, = self.ax2.plot([], [], lw=1.0)
        self.ax2.set_ylabel("RR (ms)"); self.ax2.set_xlabel("Samples"); self.ax2.grid(True, alpha=0.3)
        self.rr_txt = self.ax2.text(0.01, 0.95, "", transform=self.ax2.transAxes, va="top",
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        # α1
        if show_alpha:
            self.ax3.axhspan(0.75, 2.0, alpha=0.08, zorder=0)
            self.ax3.axhspan(0.50, 0.75, alpha=0.15, zorder=0)
            self.ax3.axhspan(0.00, 0.50, alpha=0.12, zorder=0)
            self.ax3.axhline(0.75, linestyle="--", linewidth=1)
            self.ax3.axhline(0.50, linestyle="--", linewidth=1)
            self.a_line, = self.ax3.plot([], [], lw=1.8)
            self.ax3.set_ylim(0.2, 1.6)
            self.ax3.set_ylabel("DFA α1"); self.ax3.set_xlabel("Time (s)"); self.ax3.grid(True, alpha=0.3)
            self.a_txt = self.ax3.text(0.01, 0.95, "", transform=self.ax3.transAxes, va="top",
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    def update_hr(self, y):
        if y.size == 0:
            return
        x = np.arange(y.size)
        self.hr_line.set_data(x, y)
        self.ax1.set_xlim(0, max(50, int(x[-1])))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        pad = max(3.0, 0.05 * max(1.0, y_max - y_min))
        self.ax1.set_ylim(y_min - pad, y_max + pad)
        self.hr_txt.set_text(f"HR: {y[-1]:.0f} bpm")

    def update_rr(self, rr_buf):
        if len(rr_buf) == 0:
            return
        y = np.array(list(rr_buf)[-2000:], float)
        x = np.arange(y.size)
        self.rr_line.set_data(x, y)
        self.ax2.set_xlim(0, max(50, int(x[-1])))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        pad = max(10.0, 0.05 * max(1.0, y_max - y_min))
        self.ax2.set_ylim(y_min - pad, y_max + pad)
        self.rr_txt.set_text(f"RR: {y[-1]:.0f} ms")

    def update_alpha(self, times, values):
        if self.ax3 is None or len(values) == 0:
            return
        x = np.array(list(times), float)
        y = np.array(list(values), float)
        self.a_line.set_data(x, y)
        a_curr = y[-1]
        self.a_txt.set_text(f"α1: {a_curr:.2f}")
        patch = self.a_txt.get_bbox_patch()
        if a_curr > 0.75:
            patch.set_alpha(0.20)
        elif a_curr >= 0.50:
            patch.set_alpha(0.25)
        else:
            patch.set_alpha(0.20)
        self.ax3.relim(); self.ax3.autoscale_view(scalex=True, scaley=False)

    def draw(self):
        try:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass