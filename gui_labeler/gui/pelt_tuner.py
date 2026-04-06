"""Interactive PELT parameter tuner window.

Shows a random sample of traces with live PELT re-fitting as the user
adjusts pen_mult and min_plateau sliders. An 'Apply to ALL' button
re-runs PELT on every track and rebuilds the feature matrix.
"""

import random
import threading
import time
import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from joblib import Parallel, delayed

from .. import config
from ..pelt import fit_track_pelt
from ..features import build_feature_matrix
from ..serialization import save_bundle


def _fmt_time(seconds):
    """Format seconds as MM:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class _TkTqdm:
    """tqdm-style progress bar rendered in a tk.Text widget (thread-safe).

    The bar is always at the bottom of the text widget.  Log lines
    inserted via ``PeltTunerWindow._log()`` appear *above* it.
    """

    _counter = 0
    BAR_WIDTH = 25
    FULL = "\u2588"  # █
    EMPTY = "\u2591"  # ░

    def __init__(self, text_widget, tk_window, total, desc=""):
        _TkTqdm._counter += 1
        self.tag = f"tqdm_{_TkTqdm._counter}"
        self._w = text_widget
        self._win = tk_window
        self._total = max(total, 1)
        self._desc = desc
        self._n = 0
        self._start = time.time()
        self._render()

    def update(self, n=1):
        """Advance the bar by *n* steps."""
        self._n = min(self._n + n, self._total)
        self._render()

    def close(self):
        """Finalize bar: freeze it in-place (remove tag so it persists)."""
        bar = self._format()
        tag = self.tag

        def _do():
            try:
                w = self._w
                w.configure(state=tk.NORMAL)
                ranges = w.tag_ranges(tag)
                if ranges:
                    w.delete(ranges[0], ranges[1])
                w.insert(tk.END, bar + "\n")  # no tag → permanent
                w.see(tk.END)
                w.configure(state=tk.DISABLED)
                w.update_idletasks()
            except tk.TclError:
                pass

        self._win.after(0, _do)

    def _render(self):
        bar = self._format()
        tag = self.tag

        def _do():
            try:
                w = self._w
                w.configure(state=tk.NORMAL)
                ranges = w.tag_ranges(tag)
                if ranges:
                    w.delete(ranges[0], ranges[1])
                w.insert(tk.END, bar, tag)
                w.see(tk.END)
                w.configure(state=tk.DISABLED)
                w.update_idletasks()
            except tk.TclError:
                pass

        self._win.after(0, _do)

    def _format(self):
        frac = self._n / self._total
        pct = f"{frac * 100:3.0f}%"
        filled = int(self.BAR_WIDTH * frac)
        bar = self.FULL * filled + self.EMPTY * (self.BAR_WIDTH - filled)
        elapsed = time.time() - self._start
        if self._n > 0 and elapsed > 0:
            rate = self._n / elapsed
            eta = (self._total - self._n) / rate if self._n < self._total else 0
            ts = f"[{_fmt_time(elapsed)}<{_fmt_time(eta)}, {rate:.2f}it/s]"
        else:
            ts = f"[{_fmt_time(elapsed)}<?, ?it/s]"
        return f"{self._desc}: {pct}|{bar}| {self._n}/{self._total} {ts}"


class PeltTunerWindow:
    N_SAMPLE = 6  # traces shown in the preview grid

    def __init__(self, parent_app):
        self._app = parent_app
        self._labeler = parent_app.labeler

        self._sample = []  # list of (exp, idx)
        self._figures = []
        self._canvases = []
        self._applying = False
        self._debounce_id = None

        self.win = tk.Toplevel(parent_app)
        self.win.title("PELT Parameter Tuner")
        self.win.geometry("1100x700")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._resample()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top controls bar ──────────────────────────────────────────────
        ctrl = ttk.Frame(self.win, padding=6)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        # pen_mult slider
        ttk.Label(ctrl, text="Penalty multiplier (pen×):").grid(
            row=0, column=0, sticky=tk.W
        )
        self._pen_var = tk.DoubleVar(value=1.0)
        pen_slider = ttk.Scale(
            ctrl,
            from_=0.2,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self._pen_var,
            length=260,
            command=self._on_slider,
        )
        pen_slider.grid(row=0, column=1, padx=6)
        self._pen_label = ttk.Label(ctrl, text="1.00", width=5)
        self._pen_label.grid(row=0, column=2)
        ttk.Label(ctrl, text="(↑ = fewer steps)", foreground="gray").grid(
            row=0, column=3, padx=(0, 20)
        )

        # min_plateau slider
        ttk.Label(ctrl, text="Min plateau (frames):").grid(row=1, column=0, sticky=tk.W)
        self._plat_var = tk.IntVar(value=5)
        plat_slider = ttk.Scale(
            ctrl,
            from_=2,
            to=30,
            orient=tk.HORIZONTAL,
            variable=self._plat_var,
            length=260,
            command=self._on_slider,
        )
        plat_slider.grid(row=1, column=1, padx=6)
        self._plat_label = ttk.Label(ctrl, text="5", width=5)
        self._plat_label.grid(row=1, column=2)
        ttk.Label(ctrl, text="(↑ = merge short segments)", foreground="gray").grid(
            row=1, column=3, padx=(0, 20)
        )

        # Buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=0, column=4, rowspan=2, padx=8)
        ttk.Button(btn_frame, text="Resample traces", command=self._resample).pack(
            fill=tk.X, pady=2
        )
        self._apply_btn = ttk.Button(
            btn_frame, text="Apply to ALL tracks", command=self._apply_to_all
        )
        self._apply_btn.pack(fill=tk.X, pady=2)

        # Status + progress bar
        status_frame = ttk.Frame(self.win)
        status_frame.pack(fill=tk.X, padx=4)
        self._status_var = tk.StringVar(value="Adjust sliders to preview PELT fits.")
        ttk.Label(
            status_frame, textvariable=self._status_var, relief=tk.GROOVE, padding=3
        ).pack(fill=tk.X)

        # ── Log panel (hidden until "Apply to ALL") ───────────────────────
        self._log_text = tk.Text(
            self.win,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=6,
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Menlo", 10),
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        # not packed yet — shown when the apply operation starts

        # ── Trace grid ────────────────────────────────────────────────────
        self._grid_frame = ttk.Frame(self.win)
        self._grid_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Sampling ──────────────────────────────────────────────────────────
    def _resample(self):
        all_pairs = [
            (row.exp, int(row.idx)) for _, row in self._labeler.index_df.iterrows()
        ]
        n = min(self.N_SAMPLE, len(all_pairs))
        self._sample = random.sample(all_pairs, n)
        self._refresh_plots()

    # ── Slider callback (debounced) ───────────────────────────────────────
    def _on_slider(self, _=None):
        self._pen_label.config(text=f"{self._pen_var.get():.2f}")
        self._plat_label.config(text=str(int(self._plat_var.get())))
        if self._debounce_id is not None:
            self.win.after_cancel(self._debounce_id)
        self._debounce_id = self.win.after(200, self._refresh_plots)

    # ── Plot refresh (sample only) ────────────────────────────────────────
    def _refresh_plots(self):
        pen_mult = self._pen_var.get()
        min_plateau = int(self._plat_var.get())
        self._status_var.set(
            f"Fitting sample with pen×={pen_mult:.2f}, min_plateau={min_plateau} …"
        )
        self.win.update_idletasks()

        # Clear existing figures
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
        self._canvases.clear()
        for w in self._grid_frame.winfo_children():
            w.destroy()

        ncols = 3
        for i, (exp, idx) in enumerate(self._sample):
            tr = self._labeler.all_data[exp][idx]
            nr_new = fit_track_pelt(tr, pen_mult=pen_mult, min_plateau=min_plateau)

            row, col = divmod(i, ncols)
            step_parts = "  ".join(
                f"{ch[:3]}:{nr_new[ch]['n_steps']}" for ch in config.CHANNELS
            )
            card = ttk.LabelFrame(
                self._grid_frame,
                text=f"{exp} #{idx}  [{step_parts}]",
                padding=3,
            )
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            self._grid_frame.columnconfigure(col, weight=1)
            self._grid_frame.rowconfigure(row, weight=1)

            fig = Figure(figsize=(3.8, 2.0), dpi=90)
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            t = tr["time"]
            primary_ch = config.CHANNELS[0]
            for ch in config.CHANNELS:
                color = self._labeler.channel_colors[ch]
                scale = self._labeler.calibration.get(ch, 1.0)
                target_ax = ax if ch == primary_ch else ax2
                target_ax.plot(t, tr[ch] / scale, color=color, alpha=0.45, lw=0.5)
                target_ax.plot(t, nr_new[ch]["fit"] / scale, color=color, lw=1.4)
            ax.tick_params(labelsize=6)
            ax2.tick_params(labelsize=6)
            fig.tight_layout(pad=0.3)

            self._figures.append(fig)
            canvas = FigureCanvasTkAgg(fig, master=card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._canvases.append(canvas)

        self._status_var.set(
            f"pen×={pen_mult:.2f}, min_plateau={min_plateau} — "
            f"showing {len(self._sample)} sample traces."
        )

    # ── Thread-safe log helper ────────────────────────────────────────────
    def _log(self, msg):
        """Append *msg* to the log panel, above any active tqdm bar."""

        def _append():
            try:
                w = self._log_text
                w.configure(state=tk.NORMAL)
                # Find any active tqdm bar tag → insert *before* it
                insert_pos = tk.END
                for tag_name in w.tag_names():
                    if tag_name.startswith("tqdm_"):
                        ranges = w.tag_ranges(tag_name)
                        if ranges:
                            insert_pos = str(ranges[0])
                            break
                w.insert(insert_pos, msg + "\n")
                w.see(tk.END)
                w.configure(state=tk.DISABLED)
                w.update_idletasks()
            except tk.TclError:
                pass

        self.win.after(0, _append)

    # ── Apply to all ──────────────────────────────────────────────────────
    def _apply_to_all(self):
        if self._applying:
            return
        self._applying = True
        self._apply_btn.config(state=tk.DISABLED)

        # Show and clear the log panel
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.configure(state=tk.DISABLED)
        self._log_text.pack(fill=tk.X, padx=6, pady=(2, 4), before=self._grid_frame)

        pen_mult = self._pen_var.get()
        min_plateau = int(self._plat_var.get())
        self._status_var.set(
            f"Applying pen×={pen_mult:.2f}, min_plateau={min_plateau} to ALL tracks …"
        )

        labeler = self._labeler
        experiments = labeler.experiments
        pkl_path = labeler.pkl_path

        def worker():
            self._log(f"PELT refit: pen×={pen_mult:.2f}, min_plateau={min_plateau}")

            # ── Phase 1: PELT fitting per experiment ──
            pbar = _TkTqdm(
                self._log_text,
                self.win,
                total=len(experiments),
                desc="Fitting PELT",
            )
            new_results = {}
            for exp in experiments:
                tracks = labeler.all_data[exp]
                results = Parallel(
                    n_jobs=config.N_JOBS,
                )(
                    delayed(fit_track_pelt)(
                        tr, pen_mult=pen_mult, min_plateau=min_plateau
                    )
                    for tr in tracks
                )
                new_results[exp] = results
                primary_ch = config.CHANNELS[0]
                n_steps = [r[primary_ch]["n_steps"] for r in results]
                self._log(
                    f"  {exp}: {len(tracks)} tracks — "
                    f"median {primary_ch} steps: {np.median(n_steps):.0f}"
                )
                pbar.update(1)
            pbar.close()

            # ── Phase 2: Analytical event detection ──
            pbar2 = _TkTqdm(
                self._log_text,
                self.win,
                total=1 + (1 if pkl_path else 0),
                desc="Post-processing",
            )
            pbar2.update(1)

            # ── Phase 3: Save pickle ──
            saved_mb = None
            if pkl_path:
                saved_mb = save_bundle(
                    labeler.all_data, new_results, pkl_path
                )
                self._log(f"  Pickle saved → {pkl_path} ({saved_mb:.1f} MB)")
                pbar2.update(1)
            pbar2.close()

            self._log("✓ Done — updating GUI …")
            self.win.after(
                0,
                lambda: self._on_applied(
                    new_results, pen_mult, min_plateau, saved_mb, pkl_path
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _on_applied(self, new_results, pen_mult, min_plateau, saved_mb, pkl_path):
        labeler = self._labeler

        # Update PELT results and feature matrix
        labeler.new_results = new_results
        feat_df, index_df = build_feature_matrix(
            labeler.all_data, new_results, labeler.experiments, labeler.channels
        )
        labeler.feat_df = feat_df
        labeler.index_df = index_df
        labeler._global_idx_map = {
            (row.exp, int(row.idx)): gi for gi, row in index_df.iterrows()
        }

        # Reset model since features changed
        labeler.model = None
        labeler.proba = None
        labeler.classes = None

        # Re-apply filters if a filter config is active (PELT results changed)
        if labeler.filter_config:
            n_ignored = labeler.apply_filters(labeler.filter_config)
            try:
                self._app._filter_status_var.set(
                    f"{n_ignored}/{labeler.N} traces ignored (re-applied after PELT refit)"
                )
            except Exception:
                pass

        self._applying = False
        self._apply_btn.config(state=tk.NORMAL)
        n_tracks = sum(len(labeler.all_data[e]) for e in labeler.experiments)
        pkl_msg = (
            f"  Pickle updated \u2192 {pkl_path} ({saved_mb:.1f} MB)."
            if saved_mb is not None
            else "  No pickle path \u2014 re-run --prepare to persist."
        )
        self._status_var.set(
            f"Applied pen\u00d7={pen_mult:.2f}, min_plateau={min_plateau} to "
            f"{n_tracks} tracks. "
            f"Model reset — retrain before reviewing predictions.{pkl_msg}"
        )

        # Refresh the sample plots with the new results
        self._refresh_plots()

        # Update the main window status bar
        try:
            self._app._update_status()
            log_msg = (
                f"PELT refit: pen×={pen_mult:.2f}, min_plateau={min_plateau} "
                f"\u2192 {n_tracks} tracks. Retrain the model."
            )
            if saved_mb is not None:
                log_msg += f" Pickle saved ({saved_mb:.1f} MB)."
            else:
                log_msg += " Warning: no pickle updated — re-run --prepare to persist."
            self._app._log(log_msg)
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────
    def _on_close(self):
        for fig in self._figures:
            plt.close(fig)
        self.win.destroy()
