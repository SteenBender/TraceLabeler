"""Main tkinter application window for the active-learning trace labeler."""

import json
import os
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from .. import config
from ..filters import IGNORED, FILTER_EXISTENCE, FILTER_INTENSITY, FILTER_TEMPORAL
from ..labeler import ActiveLabeler, UNLABELED
from .filter_dialog import FilterDialog
from .patch_viewer import PatchViewerWindow
from .pelt_tuner import PeltTunerWindow
from .umap_viewer import UmapViewerWindow

# Prefix shown in the review combo for system/filter labels (not user-created)
_FILTER_PREFIX = "\u2298 "  # ⊘
_FILTER_NAMES = {FILTER_EXISTENCE, FILTER_INTENSITY, FILTER_TEMPORAL}

# Candidates to prefetch / extend per batch (5 pages × 6 cards)
_PREFETCH = 30


def _is_filter_label(lbl):
    """True if *lbl* consists entirely of filter-name parts (e.g. 'existence+temporal')."""
    return bool(lbl) and all(p in _FILTER_NAMES for p in lbl.split("+"))


class _LabelerAppMethods:
    """Main application window (mixed into tk.Tk subclass by LabelerApp factory)."""

    def __init__(
        self, pkl_path=None, labels_path=None, feature_mode="pelt", cnn_device="cpu"
    ):
        tk.Tk.__init__(self)
        self.labeler = None
        self._pkl_path = pkl_path or config.PICKLE_PATH
        self._labels_path = labels_path or config.SAVE_PATH
        self._feature_mode = feature_mode
        self._cnn_device = cnn_device
        # Detect available devices (lazy import — torch may not be installed)
        try:
            from ..model import get_available_devices

            self._available_devices = get_available_devices()
        except ImportError:
            self._available_devices = ["cpu"]
        if self._cnn_device not in self._available_devices:
            self._cnn_device = self._available_devices[0]
        self.title("Active Trace Labeler")
        self.geometry("1300x900")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.current_candidates = []  # visible slice (grows on demand)
        self._candidate_pool = []  # full sorted list computed upfront
        self._pool_offset = 0  # next index to extend from pool
        self.candidate_page = 0
        self.n_per_page = 6
        self._view_mode = "random"

        # Channel configuration rows built in the startup panel
        self._ch_name_vars = []
        self._ch_color_vars = []
        self._ch_calib_vars = []
        self._ch_color_btns = []

        self._dark_theme = True
        self._show_startup_panel()

    # ══════════════════════════════════════════════════════════════════════
    # Startup panel (folder browser + channel config)
    # ══════════════════════════════════════════════════════════════════════

    def _show_startup_panel(self):
        self._startup_frame = ttk.Frame(self, padding=20)
        self._startup_frame.pack(fill=tk.BOTH, expand=True)
        f = self._startup_frame

        ttk.Label(f, text="Active Trace Labeler", font=("Helvetica", 16, "bold")).pack(
            pady=(0, 16)
        )

        # ── Data folder row ──
        folder_frame = ttk.LabelFrame(f, text="Data Folder", padding=8)
        folder_frame.pack(fill=tk.X, pady=4)

        self._folder_var = tk.StringVar(value=str(config.DATA_ROOT))
        folder_entry = ttk.Entry(folder_frame, textvariable=self._folder_var, width=60)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(folder_frame, text="Browse…", command=self._browse_folder).pack(
            side=tk.LEFT
        )
        ttk.Button(
            folder_frame, text="Detect Channels", command=self._detect_channels
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ── Channel configuration table ──
        ch_outer = ttk.LabelFrame(f, text="Channel Configuration", padding=8)
        ch_outer.pack(fill=tk.X, pady=8)

        # Header row
        hdr = ttk.Frame(ch_outer)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="#", width=3).grid(row=0, column=0, padx=4)
        ttk.Label(hdr, text="Channel Name", width=18).grid(row=0, column=1, padx=4)
        ttk.Label(hdr, text="Color", width=12).grid(row=0, column=2, padx=4)
        ttk.Label(hdr, text="Calibration (AU/mol)", width=20).grid(
            row=0, column=3, padx=4
        )

        self._ch_rows_frame = ttk.Frame(ch_outer)
        self._ch_rows_frame.pack(fill=tk.X)

        # Status label under the channel table
        self._ch_status_var = tk.StringVar(
            value="Click 'Detect Channels' to auto-fill from the selected folder."
        )
        ttk.Label(f, textvariable=self._ch_status_var, foreground="gray").pack(
            anchor=tk.W
        )

        # Pre-populate with current config channels
        self._populate_channel_rows(
            config.CHANNELS, config.CHANNEL_COLORS, config.CALIBRATION
        )

        # ── Load options ──
        load_frame = ttk.LabelFrame(f, text="Load Options", padding=8)
        load_frame.pack(fill=tk.X, pady=4)

        pkl_row = ttk.Frame(load_frame)
        pkl_row.pack(fill=tk.X, pady=2)
        ttk.Label(pkl_row, text="Cache (.pkl):", width=18).pack(side=tk.LEFT)
        self._pkl_var = tk.StringVar(value=self._pkl_path)
        ttk.Entry(pkl_row, textvariable=self._pkl_var, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(pkl_row, text="…", width=3, command=self._browse_pkl).pack(
            side=tk.LEFT
        )

        labels_row = ttk.Frame(load_frame)
        labels_row.pack(fill=tk.X, pady=2)
        ttk.Label(labels_row, text="Labels (.json):", width=18).pack(side=tk.LEFT)
        self._labels_var = tk.StringVar(value=self._labels_path)
        ttk.Entry(labels_row, textvariable=self._labels_var, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        ttk.Button(labels_row, text="…", width=3, command=self._browse_labels).pack(
            side=tk.LEFT
        )

        # ── Load button ──
        ttk.Button(f, text="Load Data & Launch", command=self._start_loading).pack(
            pady=16, ipadx=20, ipady=6
        )

        self._load_status_var = tk.StringVar()
        ttk.Label(f, textvariable=self._load_status_var, foreground="cyan").pack()

    def _browse_folder(self):
        chosen = filedialog.askdirectory(
            title="Select folder containing ExNN/ directories",
            initialdir=self._folder_var.get() or ".",
        )
        if chosen:
            self._folder_var.set(chosen)
            self._update_save_paths(chosen)
            self._detect_channels()

    def _update_save_paths(self, folder):
        """Set default pkl and labels paths to subfolders inside *folder*."""
        root = Path(folder)
        self._pkl_var.set(str(root / "results" / "trace_data.pkl"))
        self._labels_var.set(str(root / "labels" / "labels.json"))

    def _detect_channels(self):
        folder = self._folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            self._ch_status_var.set("Folder not found — enter a valid path first.")
            return
        # Discover experiment sub-directories
        exps = sorted(
            d.name
            for d in Path(folder).iterdir()
            if d.is_dir() and d.name.startswith("Ex")
        )
        if not exps:
            self._ch_status_var.set(
                "No ExNN/ sub-directories found in selected folder."
            )
            return
        config.EXPERIMENTS = exps
        n = config.detect_n_channels(folder, exps)
        if n is None:
            self._ch_status_var.set(
                "Could not read ProcessedTracks.mat — check folder."
            )
            return
        # Build default names: reuse existing names where possible
        existing = config.CHANNELS
        names = [existing[i] if i < len(existing) else f"channel_{i}" for i in range(n)]
        self._populate_channel_rows(names, config.CHANNEL_COLORS, config.CALIBRATION)
        self._ch_status_var.set(
            f"Detected {n} channel(s) across {len(exps)} experiment(s): {', '.join(exps)}"
        )

    def _populate_channel_rows(self, names, colors, calibration):
        """Rebuild the channel configuration rows."""
        for w in self._ch_rows_frame.winfo_children():
            w.destroy()
        self._ch_name_vars.clear()
        self._ch_color_vars.clear()
        self._ch_calib_vars.clear()
        self._ch_color_btns.clear()

        default_colors = config._DEFAULT_COLORS
        for i, ch in enumerate(names):
            color = colors.get(ch, default_colors[i % len(default_colors)])
            calib = calibration.get(ch, 1.0)

            row = ttk.Frame(self._ch_rows_frame)
            row.pack(fill=tk.X, pady=1)

            ttk.Label(row, text=str(i), width=3).grid(row=0, column=0, padx=4)

            name_var = tk.StringVar(value=ch)
            self._ch_name_vars.append(name_var)
            ttk.Entry(row, textvariable=name_var, width=18).grid(
                row=0, column=1, padx=4
            )

            color_var = tk.StringVar(value=color)
            self._ch_color_vars.append(color_var)
            hex_color = mcolors.to_hex(color)
            swatch = tk.Label(
                row,
                bg=hex_color,
                width=6,
                height=1,
                relief=tk.SOLID,
                borderwidth=1,
                cursor="hand2",
            )
            swatch.grid(row=0, column=2, padx=4)
            swatch.bind("<Button-1>", lambda e, idx=i: self._pick_color(idx) or "break")
            self._ch_color_btns.append(swatch)

            calib_var = tk.StringVar(value=str(calib))
            self._ch_calib_vars.append(calib_var)
            ttk.Entry(row, textvariable=calib_var, width=12).grid(
                row=0, column=3, padx=4
            )

    def _open_color_picker(self, current_hex: str) -> "str | None":
        """Open the custom colour chooser dialog and return the chosen hex, or None."""
        dlg = tk.Toplevel(self)
        dlg.title("Pick colour")
        dlg.resizable(False, False)
        dlg.grab_set()

        chosen = [None]

        # Preset palette – ordered by hue (red → yellow → green → cyan → blue → purple),
        # then pastels, then neutrals.
        palette = [
            # Row 1: vivid primaries & secondaries
            "#ff0000",
            "#ff4400",
            "#ff8800",
            "#ffbb00",
            "#ffff00",
            "#88ff00",
            "#00ff00",
            "#00ffaa",
            "#00ffff",
            "#00aaff",
            # Row 2: blues, purples, pinks
            "#0044ff",
            "#0000ff",
            "#4400ff",
            "#8800ff",
            "#bb00ff",
            "#ff00ff",
            "#ff0088",
            "#ff0044",
            "#ff6688",
            "#ffaacc",
            # Row 3: pastels
            "#ffcccc",
            "#ffddbf",
            "#ffffcc",
            "#ccffcc",
            "#ccffff",
            "#cce0ff",
            "#ccccff",
            "#e8ccff",
            "#ffccee",
            "#ffffff",
            # Row 4: darks & neutrals
            "#800000",
            "#884400",
            "#808000",
            "#006600",
            "#006666",
            "#000080",
            "#440088",
            "#660044",
            "#a0a0a0",
            "#404040",
        ]

        grid = ttk.Frame(dlg, padding=8)
        grid.pack()
        cols = 10
        for i_c, c in enumerate(palette):
            btn = tk.Label(
                grid,
                bg=c,
                width=3,
                height=1,
                relief=tk.RAISED,
                borderwidth=1,
                cursor="hand2",
            )
            btn.grid(row=i_c // cols, column=i_c % cols, padx=1, pady=1)
            btn.bind("<Button-1>", lambda e, col=c: _select(col))

        entry_frame = ttk.Frame(dlg, padding=(8, 4))
        entry_frame.pack(fill=tk.X)
        ttk.Label(entry_frame, text="Hex:").pack(side=tk.LEFT)
        hex_var = tk.StringVar(value=current_hex)
        ttk.Entry(entry_frame, textvariable=hex_var, width=10).pack(
            side=tk.LEFT, padx=4
        )

        preview = tk.Label(
            entry_frame,
            bg=current_hex,
            width=4,
            height=1,
            relief=tk.SOLID,
            borderwidth=1,
        )
        preview.pack(side=tk.LEFT, padx=4)

        def _update_preview(*_args):
            try:
                preview.configure(bg=mcolors.to_hex(hex_var.get()))
            except ValueError:
                pass

        hex_var.trace_add("write", _update_preview)

        def _select(col):
            chosen[0] = col
            dlg.destroy()

        def _confirm():
            try:
                chosen[0] = mcolors.to_hex(hex_var.get())
            except ValueError:
                chosen[0] = hex_var.get()
            dlg.destroy()

        btn_frame = ttk.Frame(dlg, padding=8)
        btn_frame.pack()
        ttk.Button(btn_frame, text="OK", command=_confirm).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(
            side=tk.LEFT, padx=4
        )

        dlg.wait_window()
        return chosen[0]

    def _pick_color(self, idx):
        """Colour picker for the startup-panel channel config rows."""
        current = self._ch_color_vars[idx].get()
        try:
            hex_current = mcolors.to_hex(current)
        except ValueError:
            hex_current = "#ffffff"

        hex_color = self._open_color_picker(hex_current)
        if hex_color:
            self._ch_color_vars[idx].set(hex_color)
            self._ch_color_btns[idx].configure(bg=hex_color)

    def _pick_legend_color(self, ch, swatch):
        """Colour picker for the live channel legend; updates plots immediately."""
        current = mcolors.to_hex(self.labeler.channel_colors[ch])
        hex_color = self._open_color_picker(current)
        if hex_color:
            self.labeler.channel_colors[ch] = hex_color
            swatch.configure(bg=hex_color)
            if self.card_widgets:
                self._render_page()

    def _browse_pkl(self):
        path = filedialog.asksaveasfilename(
            title="Cache file (.pkl)",
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")],
            initialfile=os.path.basename(self._pkl_var.get()),
            initialdir=os.path.dirname(self._pkl_var.get()) or ".",
        )
        if path:
            self._pkl_var.set(path)

    def _browse_labels(self):
        path = filedialog.asksaveasfilename(
            title="Labels file (.json)",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile=os.path.basename(self._labels_var.get()),
            initialdir=os.path.dirname(self._labels_var.get()) or ".",
        )
        if path:
            self._labels_var.set(path)

    def _collect_channel_config(self):
        """Read the channel rows and call config.configure_channels()."""
        names = [
            v.get().strip() or f"channel_{i}" for i, v in enumerate(self._ch_name_vars)
        ]
        colors = {n: self._ch_color_vars[i].get() for i, n in enumerate(names)}
        calibration = {}
        for i, n in enumerate(names):
            try:
                calibration[n] = float(self._ch_calib_vars[i].get())
            except ValueError:
                calibration[n] = 1.0
        config.configure_channels(names, colors, calibration)

    def _start_loading(self):
        folder = self._folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("Error", "Please select a valid data folder.")
            return
        if not self._ch_name_vars:
            messagebox.showerror(
                "Error", "No channels configured. Click 'Detect Channels' first."
            )
            return

        config.DATA_ROOT = Path(folder)
        config.TIFF_ROOT = Path(folder)
        self._pkl_path = self._pkl_var.get()
        self._labels_path = self._labels_var.get()
        self._collect_channel_config()

        self._load_status_var.set("Loading… please wait.")
        self.update_idletasks()
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        """Run in background thread — load/prepare data then hand off to UI thread."""
        from ..data_loader import _load_and_pelt
        from ..serialization import load_prepared_data, save_bundle
        from ..gui.splash import LoadingSplash

        try:
            pkl = self._pkl_path
            if os.path.exists(pkl):
                self.after(0, lambda: self._load_status_var.set("Loading cached data…"))
                all_data, pelt_results, stored_channels = load_prepared_data(pkl)
                # Sync config to pkl when channel counts differ (e.g. 2-channel data
                # loaded while config still defaults to 3 channels).
                if stored_channels != config.CHANNELS:
                    if len(stored_channels) != len(config.CHANNELS):
                        # Trust the pkl — update config to match what was actually saved.
                        new_colors = {
                            ch: config.CHANNEL_COLORS.get(
                                ch, config._DEFAULT_COLORS[i % len(config._DEFAULT_COLORS)]
                            )
                            for i, ch in enumerate(stored_channels)
                        }
                        new_calib = {
                            ch: config.CALIBRATION.get(ch, 1.0)
                            for ch in stored_channels
                        }
                        config.configure_channels(stored_channels, new_colors, new_calib)
                    else:
                        # Same count but different names — rename data keys to match
                        # the names the user typed in the startup panel.
                        rename_map = {
                            old: new
                            for old, new in zip(stored_channels, config.CHANNELS)
                            if old != new
                        }
                        for exp in all_data:
                            for tr in all_data[exp]:
                                for old, new in rename_map.items():
                                    if old in tr:
                                        tr[new] = tr.pop(old)
                                    for suffix in (
                                        "_fit",
                                        "_numSteps",
                                        "_stepStart",
                                        "_stepSize",
                                    ):
                                        if f"{old}{suffix}" in tr:
                                            tr[f"{new}{suffix}"] = tr.pop(
                                                f"{old}{suffix}"
                                            )
                            for nr in pelt_results[exp]:
                                for old, new in rename_map.items():
                                    if old in nr:
                                        nr[new] = nr.pop(old)
            else:
                self.after(
                    0,
                    lambda: self._load_status_var.set(
                        "Running PELT fitting (may take a few minutes)…"
                    ),
                )
                all_data, pelt_results = _load_and_pelt()
                os.makedirs(os.path.dirname(os.path.abspath(pkl)) or ".", exist_ok=True)
                save_bundle(all_data, pelt_results, pkl)

            self.after(0, lambda: self._finish_loading(all_data, pelt_results))
        except Exception as exc:
            import traceback

            self.after(0, lambda: self._load_status_var.set(f"Error: {exc}"))
            traceback.print_exc()

    def _finish_loading(self, all_data, pelt_results):
        from ..labeler import ActiveLabeler

        labels_path = self._labels_path
        os.makedirs(os.path.dirname(os.path.abspath(labels_path)) or ".", exist_ok=True)

        labeler = ActiveLabeler(
            all_data,
            pelt_results,
            config.CHANNELS,
            config.CHANNEL_COLORS,
            experiments=config.EXPERIMENTS,
            save_path=labels_path,
            calibration=config.CALIBRATION,
            feature_mode=self._feature_mode,
            cnn_device=self._cnn_device,
        )
        labeler.pkl_path = self._pkl_path if os.path.exists(self._pkl_path) else None

        if os.path.exists(labels_path):
            try:
                n = labeler.load(labels_path)
                print(f"  Auto-loaded {n} labels from {labels_path}")
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"  Warning: could not load labels: {exc}")

        self.labeler = labeler
        self._startup_frame.destroy()
        self._build_ui()
        # Restore filter status if sub_labels were loaded
        if labeler.n_ignored > 0:
            self._filter_status_var.set(
                f"{labeler.n_ignored}/{labeler.N} traces ignored"
            )
            # Auto-enable "Show ignored traces" so filter labels appear in the combo
            self._show_ignored_var.set(True)
        self._update_status()

    # ══════════════════════════════════════════════════════════════════════
    # Main labeler UI — only shown after data is loaded
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Top status bar
        self.status_var = tk.StringVar(value="Active Labeler")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            font=("Helvetica", 11, "bold"),
            relief=tk.GROOVE,
            padding=4,
        )
        status_bar.pack(fill=tk.X, padx=4, pady=(4, 0))

        # ── Left panel (labeling workflow) ──
        ctrl = ttk.LabelFrame(self, text="Controls", padding=6)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

        # Add label
        ttk.Label(ctrl, text="New label:").pack(anchor=tk.W)
        self.new_label_var = tk.StringVar()
        entry = ttk.Entry(ctrl, textvariable=self.new_label_var, width=22)
        entry.pack(fill=tk.X)
        entry.bind("<Return>", lambda e: self._add_label())
        ttk.Button(ctrl, text="Add Label", command=self._add_label).pack(
            fill=tk.X, pady=(2, 8)
        )

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Feature mode selector + Device + Train
        model_frame = ttk.Frame(ctrl)
        model_frame.pack(fill=tk.X)
        # Features
        feat_frame = ttk.Frame(model_frame)
        feat_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Label(feat_frame, text="Features:").pack(anchor=tk.W)
        self._feature_var = tk.StringVar(
            value="Pretrained" if self._feature_mode == "pretrained" else "PELT"
        )
        feature_combo = ttk.Combobox(
            feat_frame,
            textvariable=self._feature_var,
            values=["PELT", "Pretrained"],
            state="readonly",
            width=10,
        )
        feature_combo.pack(fill=tk.X)
        feature_combo.bind("<<ComboboxSelected>>", self._on_feature_changed)
        # Device
        dev_frame = ttk.Frame(model_frame)
        dev_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        ttk.Label(dev_frame, text="Device:").pack(anchor=tk.W)
        self._device_var = tk.StringVar(value=self._cnn_device)
        self._device_combo = ttk.Combobox(
            dev_frame,
            textvariable=self._device_var,
            values=self._available_devices,
            state="readonly",
            width=8,
        )
        self._device_combo.pack(fill=tk.X)
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_changed)

        self._pretrain_btn = ttk.Button(
            ctrl, text="Pretrain Encoder", command=self._pretrain_cnn
        )
        self._train_btn = ttk.Button(ctrl, text="Train Model", command=self._train)
        if self._feature_mode == "pretrained":
            self._pretrain_btn.pack(fill=tk.X, pady=2)
        self._train_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Review predictions
        ttk.Label(ctrl, text="Review label:").pack(anchor=tk.W)
        self.review_label_var = tk.StringVar()
        self.review_combo = ttk.Combobox(
            ctrl, textvariable=self.review_label_var, state="readonly", width=20
        )
        self.review_combo.pack(fill=tk.X)
        pred_row = ttk.Frame(ctrl)
        pred_row.pack(fill=tk.X, pady=2)
        ttk.Button(pred_row, text="Show Top", command=self._show_predictions).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(pred_row, text="Show Low", command=self._show_low_confidence).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(ctrl, text="Show Labeled", command=self._show_labeled).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(ctrl, text="Show Random Traces", command=self._show_random).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(ctrl, text="Delete Class", command=self._delete_label_class).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Navigation
        nav = ttk.Frame(ctrl)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="◀ Prev", command=self._prev_page).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(nav, text="Next ▶", command=self._next_page).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(ctrl, text="Confirm ALL on Page", command=self._confirm_all).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(ctrl, text="Show Patches", command=self._show_patches).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Save / Load
        save_load_row = ttk.Frame(ctrl)
        save_load_row.pack(fill=tk.X, pady=2)
        ttk.Button(save_load_row, text="Save Labels", command=self._save).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(save_load_row, text="Load Labels", command=self._load).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )

        # ── Gallery frame (created now, packed AFTER right panel so right panel
        #    gets its space before the expand=True gallery claims the rest) ──
        self.gallery_frame = ttk.Frame(self)

        # Placeholder label
        self.placeholder = ttk.Label(
            self.gallery_frame,
            text=(
                "Welcome! Use the controls on the left to:\n\n"
                "1.  Add label classes (e.g. 'hsc70_excess', 'balanced')\n"
                "2.  Label some traces via Show Random\n"
                "3.  Train the model\n"
                "4.  Review top/low-confidence predictions per label\n\n"
                "Traces will appear here."
            ),
            font=("Helvetica", 12),
            justify=tk.CENTER,
        )
        self.placeholder.pack(expand=True)

        # List to hold per-card widgets so we can destroy them
        self.card_widgets = []
        self._card_figures = []  # track matplotlib figures for cleanup

        # ── Right panel (tools & settings) — packed BEFORE gallery so it
        #    claims its natural width before gallery's expand=True runs ──
        right = ttk.LabelFrame(self, text="Tools", padding=6)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)

        # ── Filters ──
        filter_frame = ttk.LabelFrame(right, text="Filters", padding=4)
        filter_frame.pack(fill=tk.X, pady=2)
        ttk.Button(
            filter_frame, text="Configure Filters...", command=self._open_filter_dialog
        ).pack(fill=tk.X, pady=2)
        self._show_ignored_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            filter_frame,
            text="Show ignored traces",
            variable=self._show_ignored_var,
            command=self._on_show_ignored_toggled,
        ).pack(anchor=tk.W)
        self._train_ignored_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            filter_frame,
            text="Include ignored in training",
            variable=self._train_ignored_var,
            command=self._on_train_ignored_toggled,
        ).pack(anchor=tk.W)
        self._filter_status_var = tk.StringVar(value="No filters active")
        ttk.Label(
            filter_frame,
            textvariable=self._filter_status_var,
            foreground="gray",
            font=("Helvetica", 8),
        ).pack(anchor=tk.W)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # PELT Tuner
        ttk.Button(right, text="PELT Tuner", command=self._show_pelt_tuner).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(right, text="UMAP Explorer", command=self._show_umap_explorer).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ── Plot limits ──
        limits_frame = ttk.LabelFrame(right, text="Plot Limits", padding=4)
        limits_frame.pack(fill=tk.X, pady=2)

        self._xlim_min_var  = tk.StringVar()
        self._xlim_max_var  = tk.StringVar()
        self._ylim_min_var  = tk.StringVar()
        self._ylim_max_var  = tk.StringVar()
        self._ylim2_min_var = tk.StringVar()
        self._ylim2_max_var = tk.StringVar()

        lim_grid = ttk.Frame(limits_frame)
        lim_grid.pack(fill=tk.X)
        for row_i, (lbl, lo_var, hi_var) in enumerate([
            ("X:",   self._xlim_min_var,  self._xlim_max_var),
            ("Y_l:", self._ylim_min_var,  self._ylim_max_var),
            ("Y_r:", self._ylim2_min_var, self._ylim2_max_var),
        ]):
            ttk.Label(lim_grid, text=lbl, anchor=tk.E, width=4).grid(
                row=row_i, column=0, sticky=tk.E, padx=(0, 4), pady=1
            )
            ttk.Entry(lim_grid, textvariable=lo_var, width=6).grid(
                row=row_i, column=1, padx=1, pady=1
            )
            ttk.Label(lim_grid, text="–").grid(row=row_i, column=2, padx=2)
            ttk.Entry(lim_grid, textvariable=hi_var, width=6).grid(
                row=row_i, column=3, padx=1, pady=1
            )

        ttk.Label(limits_frame, text="Leave blank for auto-scale", foreground="gray",
                  font=("Helvetica", 8)).pack(anchor=tk.W, pady=(2, 0))
        ttk.Button(
            limits_frame, text="Apply", command=self._apply_plot_limits
        ).pack(fill=tk.X, pady=(2, 0))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Diagnostics
        ttk.Button(right, text="Diagnostics", command=self._show_diagnostics).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            right, text="Save Predictions (CSV)", command=self._save_predictions
        ).pack(fill=tk.X, pady=2)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Theme toggle
        self._theme_btn_var = tk.StringVar(value="Theme: Dark")
        ttk.Button(
            right, textvariable=self._theme_btn_var, command=self._toggle_theme
        ).pack(fill=tk.X, pady=2)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ── Channel legend ──
        legend_frame = ttk.LabelFrame(right, text="Channel Legend", padding=4)
        legend_frame.pack(fill=tk.X, pady=2)
        for ch in self.labeler.channels:
            hex_color = mcolors.to_hex(self.labeler.channel_colors[ch])
            row = ttk.Frame(legend_frame)
            row.pack(fill=tk.X, pady=1)
            swatch = tk.Label(
                row,
                bg=hex_color,
                width=2,
                height=1,
                relief=tk.SOLID,
                borderwidth=1,
                cursor="hand2",
            )
            swatch.pack(side=tk.LEFT, padx=(0, 4))
            swatch.bind(
                "<Button-1>",
                lambda e, c=ch, s=swatch: self._pick_legend_color(c, s),
            )
            ttk.Label(row, text=ch).pack(side=tk.LEFT)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ── Log panel ──
        ttk.Label(right, text="Log:").pack(anchor=tk.W, pady=(4, 0))
        self.log_text = tk.Text(
            right,
            width=28,
            height=12,
            font=("Courier", 9),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ── Gallery packed last so it fills remaining space ──
        self.gallery_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Logging ──────────────────────────────────────────────────────────
    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ── Status ───────────────────────────────────────────────────────────
    def _update_status(self):
        lc = self.labeler.label_counts()
        counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(lc.items())) or "(none)"
        hist = ""
        if self.labeler.history:
            h = self.labeler.history[-1]
            hist = f" | Train={h['train_acc']:.2f}, CV={h['cv_acc']:.2f}"
        feat_tag = "Pretrained" if self._feature_mode == "pretrained" else "PELT"
        ignored_str = ""
        if self.labeler.n_ignored > 0:
            ignored_str = f"  |  Ignored: {self.labeler.n_ignored}"
        self.status_var.set(
            f"[{feat_tag}] "
            f"Round {self.labeler.round_num}  |  "
            f"Labeled: {self.labeler.n_labeled}/{self.labeler.N}{ignored_str}  |  "
            f"[{counts_str}]{hist}"
        )
        # Update review combo — filter labels are shown only when show_ignored is on,
        # and are prefixed with ⊘ to visually distinguish them from user labels.
        all_labels = sorted(self.labeler.label_set) if self.labeler.label_set else []
        if self._show_ignored_var.get():
            opts = [
                (_FILTER_PREFIX + l if _is_filter_label(l) else l) for l in all_labels
            ]
        else:
            opts = [l for l in all_labels if not _is_filter_label(l)]
        self.review_combo["values"] = opts
        if opts and self.review_label_var.get() not in opts:
            self.review_label_var.set(opts[0])

    def _get_review_label(self):
        """Return the raw stored label from the review combo (strips ⊘ prefix)."""
        val = self.review_label_var.get()
        return val[len(_FILTER_PREFIX) :] if val.startswith(_FILTER_PREFIX) else val

    def _on_show_ignored_toggled(self):
        self._update_status()
        if not self._show_ignored_var.get() and self.current_candidates:
            # Remove ignored traces from the live pools so they don't appear on page-flip
            self._candidate_pool = [
                c for c in self._candidate_pool
                if self.labeler.get_sub_label(c["exp"], c["idx"]) != IGNORED
            ]
            self.current_candidates = [
                c for c in self.current_candidates
                if self.labeler.get_sub_label(c["exp"], c["idx"]) != IGNORED
            ]
            max_page = max(0, (len(self.current_candidates) - 1) // self.n_per_page)
            self.candidate_page = min(self.candidate_page, max_page)
            self._render_page()
        elif self._show_ignored_var.get() and self.card_widgets:
            self._render_page()

    def _apply_plot_limits(self):
        if self.card_widgets:
            self._render_page()

    # ── Actions ──────────────────────────────────────────────────────────
    def _add_label(self):
        name = self.new_label_var.get().strip()
        if not name:
            return
        self.labeler.label_set.add(name)
        self.new_label_var.set("")
        self._update_status()
        self._log(f"Added label: '{name}'")

    def _on_feature_changed(self, event=None):
        choice = self._feature_var.get()
        new_mode = "pretrained" if choice == "Pretrained" else "pelt"
        self._feature_mode = new_mode
        self.labeler.feature_mode = new_mode
        if new_mode == "pretrained":
            self._pretrain_btn.pack(fill=tk.X, pady=2, before=self._train_btn)
        else:
            self._pretrain_btn.pack_forget()
        self._log(f"Switched to {choice} features")
        self._update_status()

    def _on_device_changed(self, event=None):
        self._cnn_device = self._device_var.get()
        self.labeler.cnn_device = self._cnn_device
        self._log(f"Device set to {self._cnn_device}")

    def _pretrain_cnn(self):
        self._log("Pretraining CNN (masked autoencoder) …")
        self.update_idletasks()

        def _run():
            msg = self.labeler.pretrain_cnn()
            self.after(0, lambda: self._log(msg))

        threading.Thread(target=_run, daemon=True).start()

    def _train(self):
        self._log("Training …")
        self.update_idletasks()
        result, msg = self.labeler.train()
        if result is None and msg:
            messagebox.showwarning("Train", msg)
        if msg:
            self._log(msg)
        self._update_status()

    def _show_predictions(self):
        target = self._get_review_label()
        if not target:
            return
        if self.labeler.proba is None:
            messagebox.showinfo("Review", "Train the model first.")
            return
        show_ignored = self._show_ignored_var.get()
        pool = self.labeler.get_top_predictions(
            target, n=None, unlabeled_only=True, show_ignored=show_ignored
        )
        if not pool:
            self._log(f"No candidates for '{target}'")
            return
        self._candidate_pool = pool
        self._pool_offset = min(_PREFETCH, len(pool))
        self.current_candidates = pool[: self._pool_offset]
        self._view_mode = "top"
        self.candidate_page = 0
        self._render_page()

    def _show_low_confidence(self):
        target = self._get_review_label()
        if not target:
            return
        if self.labeler.proba is None:
            messagebox.showinfo("Review", "Train the model first.")
            return
        show_ignored = self._show_ignored_var.get()
        pool = self.labeler.get_low_confidence_predictions(
            target, n=None, unlabeled_only=True, show_ignored=show_ignored
        )
        if not pool:
            self._log(f"No low-confidence candidates for '{target}'")
            return
        self._candidate_pool = pool
        self._pool_offset = min(_PREFETCH, len(pool))
        self.current_candidates = pool[: self._pool_offset]
        self._view_mode = "low_conf"
        self.candidate_page = 0
        self._render_page()

    def _show_random(self):
        show_ignored = self._show_ignored_var.get()
        pairs = [(row.exp, int(row.idx)) for _, row in self.labeler.index_df.iterrows()]
        if not show_ignored:
            pairs = [p for p in pairs if self.labeler.get_sub_label(*p) != IGNORED]
        if not pairs:
            self._log("No eligible traces to show.")
            return
        random.shuffle(pairs)
        pool = [
            {
                "exp": e,
                "idx": i,
                "prob": None,
                "current_label": self.labeler.get_label(e, i),
            }
            for e, i in pairs
        ]
        self._candidate_pool = pool
        self._pool_offset = min(_PREFETCH, len(pool))
        self.current_candidates = pool[: self._pool_offset]
        self._view_mode = "random"
        self.candidate_page = 0
        self._render_page()

    def _show_labeled(self):
        """Show all traces labeled with the currently selected review label,
        sorted by model probability (descending) when a model is trained."""
        target = self._get_review_label()
        if not target:
            return
        pairs = [
            (row.exp, int(row.idx))
            for _, row in self.labeler.index_df.iterrows()
            if self.labeler.get_label(row.exp, int(row.idx)) == target
        ]
        if not pairs:
            self._log(f"No traces labeled as '{target}'")
            return
        # Sort by probability descending when model is available
        if self.labeler.proba is not None and target in list(self.labeler.classes):
            label_idx = list(self.labeler.classes).index(target)
            pool = []
            for exp, idx in pairs:
                gi = self.labeler._global_idx(exp, idx)
                prob = (
                    float(self.labeler.proba[gi, label_idx]) if gi is not None else 0.0
                )
                pool.append(
                    {"exp": exp, "idx": idx, "prob": prob, "current_label": target}
                )
            pool.sort(key=lambda x: -x["prob"])
        else:
            pool = [
                {"exp": e, "idx": i, "prob": None, "current_label": target}
                for e, i in pairs
            ]
        self._candidate_pool = pool
        self._pool_offset = min(_PREFETCH, len(pool))
        self.current_candidates = pool[: self._pool_offset]
        self._view_mode = "labeled"
        self.candidate_page = 0
        self._render_page()
        self._log(f"Showing {len(pool)} labeled traces for '{target}'")

    def _show_ignored_traces(self):
        """Show all traces marked as ignored by the active filters."""
        pairs = [
            (row.exp, int(row.idx))
            for _, row in self.labeler.index_df.iterrows()
            if self.labeler.get_sub_label(row.exp, int(row.idx)) == IGNORED
        ]
        if not pairs:
            self._log("No ignored traces to show.")
            return
        self.current_candidates = [
            {
                "exp": e,
                "idx": i,
                "prob": None,
                "current_label": self.labeler.get_label(e, i),
            }
            for e, i in pairs
        ]
        self._view_mode = "random"
        self.candidate_page = 0
        self._render_page()
        self._log(f"Showing {len(pairs)} ignored traces")

    def _show_patches(self):
        if not self.current_candidates:
            messagebox.showinfo(
                "Patch Viewer",
                "No traces loaded. Use Show Top Predictions, "
                "Show Low Confidence, or Show Random Traces first.",
            )
            return
        PatchViewerWindow(self, self.current_candidates, start_page=self.candidate_page)

    def _show_pelt_tuner(self):
        PeltTunerWindow(self)

    def _show_umap_explorer(self):
        UmapViewerWindow(self)

    # ── Filters ──────────────────────────────────────────────────────────
    def _open_filter_dialog(self):
        dlg = FilterDialog(self, self.labeler.channels, self.labeler.filter_config)
        self.wait_window(dlg)
        result = dlg.get_result()
        if result is None:
            return  # cancelled
        if result == "clear":
            self.labeler.clear_filters()
            self._filter_status_var.set("No filters active")
            self._log("Filters cleared")
            self._update_status()
            return
        n_ignored = self.labeler.apply_filters(result)
        n_total = self.labeler.N
        self._filter_status_var.set(f"{n_ignored}/{n_total} traces ignored")
        self._log(f"Filters applied: {n_ignored}/{n_total} traces marked ignored")
        self._update_status()

    def _on_train_ignored_toggled(self):
        self.labeler.include_ignored_in_training = self._train_ignored_var.get()

    def _prev_page(self):
        if self.candidate_page > 0:
            self.candidate_page -= 1
            self._render_page()

    def _next_page(self):
        max_page = max(0, (len(self.current_candidates) - 1) // self.n_per_page)
        if self.candidate_page < max_page:
            self.candidate_page += 1
            self._render_page()
        elif self._pool_offset < len(self._candidate_pool):
            # Reached end of visible slice — extend from pool
            next_batch = self._candidate_pool[
                self._pool_offset : self._pool_offset + _PREFETCH
            ]
            self.current_candidates.extend(next_batch)
            self._pool_offset += len(next_batch)
            self.candidate_page += 1
            self._render_page()

    def _confirm_all(self):
        target = self._get_review_label()
        if not target:
            return
        start = self.candidate_page * self.n_per_page
        page_cands = self.current_candidates[start : start + self.n_per_page]
        for c in page_cands:
            self.labeler.set_label(c["exp"], c["idx"], target)
        self._update_status()
        self._log(f"Confirmed {len(page_cands)} as '{target}'")
        self._render_page()

    # ── Gallery rendering ────────────────────────────────────────────────
    def _toggle_theme(self):
        self._dark_theme = not self._dark_theme
        if self._dark_theme:
            plt.style.use("dark_background")
            self._theme_btn_var.set("Theme: Dark")
        else:
            plt.style.use("default")
            self._theme_btn_var.set("Theme: Light")
        if self.card_widgets:
            self._render_page()

    def _clear_gallery(self):
        self.placeholder.pack_forget() if self.placeholder.winfo_manager() else None
        # Close matplotlib figures to prevent memory leaks
        for fig in self._card_figures:
            plt.close(fig)
        self._card_figures.clear()
        for w in self.card_widgets:
            w.destroy()
        self.card_widgets.clear()

    def _render_page(self):
        self._clear_gallery()

        is_random = self._view_mode in ("random", "labeled")
        target = None if self._view_mode == "random" else self._get_review_label()
        start = self.candidate_page * self.n_per_page
        page = self.current_candidates[start : start + self.n_per_page]
        n_shown = len(self.current_candidates)
        n_pool = len(self._candidate_pool)
        max_page = max(0, (n_shown - 1) // self.n_per_page)
        # Show total pool size so user knows how many traces exist
        total_str = f"{n_pool}" if n_pool > n_shown else f"{n_shown}"
        page_str = (
            f"Page {self.candidate_page + 1}/{max_page + 1}"
            f"  ({n_shown} loaded / {total_str} total)"
        )

        if self._view_mode == "random":
            header_text = f"Random traces  —  {page_str}"
        elif self._view_mode == "labeled":
            header_text = f"Labeled '{target}'  —  {page_str}"
        elif self._view_mode == "low_conf":
            header_text = f"Low-confidence predictions for '{target}'  —  {page_str}"
        else:
            header_text = f"Top predictions for '{target}'  —  {page_str}"
        header = ttk.Label(
            self.gallery_frame,
            text=header_text,
            font=("Helvetica", 10, "bold"),
        )
        header.pack(anchor=tk.W, pady=(0, 4))
        self.card_widgets.append(header)

        # Arrange cards in a 2-column grid inside a scrollable canvas
        container = ttk.Frame(self.gallery_frame)
        container.pack(fill=tk.BOTH, expand=True)
        self.card_widgets.append(container)

        ncols = 2
        for i, cand in enumerate(page):
            row, col = divmod(i, ncols)
            card = self._build_card(container, cand, target, start + i)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            container.columnconfigure(col, weight=1)

    def _build_card(self, parent, cand, target_label, abs_idx):
        """Build a card frame with matplotlib trace + buttons.

        target_label=None means random-browse mode: show a label picker instead
        of a fixed confirm button.
        """
        exp, idx, prob = cand["exp"], cand["idx"], cand["prob"]
        is_random = prob is None

        title = f"{exp} #{idx}" if is_random else f"{exp} #{idx}  P={prob:.3f}"
        if self.labeler.get_sub_label(exp, idx) == IGNORED:
            title += f"  {_FILTER_PREFIX.strip()}IGNORED"
        card = ttk.LabelFrame(parent, text=title, padding=4)

        # Matplotlib figure embedded in tk
        fig = Figure(figsize=(5.2, 2.0), dpi=90)
        ax = fig.add_subplot(111)
        title_extra = "" if is_random else f"  P({target_label})={prob:.2f}"
        self.labeler.plot_trace(exp, idx, ax=ax, title_extra=title_extra)
        # fig.axes[1] is the twinx secondary y-axis created inside plot_trace
        ax2 = fig.axes[1] if len(fig.axes) > 1 else None
        self._apply_axis_limits(ax, ax2)
        fig.tight_layout(pad=0.4)
        self._card_figures.append(fig)
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button row — grid layout so all items share equal width
        btn_row = ttk.Frame(card)
        btn_row.pack(fill=tk.X, pady=(2, 0))

        all_labels = sorted(self.labeler.label_set)

        if is_random:
            # 4 equal columns: [combo] [Label] [✕ Unlabel] [Save PDF]
            label_var = tk.StringVar(value=all_labels[0] if all_labels else "")
            ttk.Combobox(
                btn_row,
                textvariable=label_var,
                values=all_labels,
                state="readonly",
                width=6,
            ).grid(row=0, column=0, padx=2, sticky="ew")
            ttk.Button(
                btn_row,
                text="Label",
                command=lambda e=exp, i=idx, rv=label_var: self._confirm_one(
                    e, i, rv.get(), card
                ),
            ).grid(row=0, column=1, padx=2, sticky="ew")
            ttk.Button(
                btn_row,
                text="✕ Unlabel",
                command=lambda e=exp, i=idx: self._unlabel_one(e, i, card),
            ).grid(row=0, column=2, padx=2, sticky="ew")
            ttk.Button(
                btn_row,
                text="Save PDF",
                command=lambda f=fig, t=title, e=exp, i=idx: self._save_trace_pdf(
                    f, t, e, i
                ),
            ).grid(row=0, column=3, padx=2, sticky="ew")
            for col in range(4):
                btn_row.columnconfigure(col, weight=1)
        else:
            other_labels = [l for l in all_labels if l != target_label]
            if other_labels:
                # 5 equal columns: [✓ label] [combo] [Assign] [✕ Unlabel] [Save PDF]
                reassign_var = tk.StringVar(value=other_labels[0])
                ttk.Button(
                    btn_row,
                    text=f"✓ {target_label}",
                    command=lambda e=exp, i=idx, tl=target_label: self._confirm_one(
                        e, i, tl, card
                    ),
                ).grid(row=0, column=0, padx=2, sticky="ew")
                ttk.Combobox(
                    btn_row,
                    textvariable=reassign_var,
                    values=other_labels,
                    state="readonly",
                    width=5,
                ).grid(row=0, column=1, padx=2, sticky="ew")
                ttk.Button(
                    btn_row,
                    text="Assign",
                    command=lambda e=exp, i=idx, rv=reassign_var: self._reassign_one(
                        e, i, rv.get(), card
                    ),
                ).grid(row=0, column=2, padx=2, sticky="ew")
                ttk.Button(
                    btn_row,
                    text="✕ Unlabel",
                    command=lambda e=exp, i=idx: self._unlabel_one(e, i, card),
                ).grid(row=0, column=3, padx=2, sticky="ew")
                ttk.Button(
                    btn_row,
                    text="Save PDF",
                    command=lambda f=fig, t=title, e=exp, i=idx: self._save_trace_pdf(
                        f, t, e, i
                    ),
                ).grid(row=0, column=4, padx=2, sticky="ew")
                for col in range(5):
                    btn_row.columnconfigure(col, weight=1)
            else:
                # 3 equal columns: [✓ label] [✕ Unlabel] [Save PDF]
                ttk.Button(
                    btn_row,
                    text=f"✓ {target_label}",
                    command=lambda e=exp, i=idx, tl=target_label: self._confirm_one(
                        e, i, tl, card
                    ),
                ).grid(row=0, column=0, padx=2, sticky="ew")
                ttk.Button(
                    btn_row,
                    text="✕ Unlabel",
                    command=lambda e=exp, i=idx: self._unlabel_one(e, i, card),
                ).grid(row=0, column=1, padx=2, sticky="ew")
                ttk.Button(
                    btn_row,
                    text="Save PDF",
                    command=lambda f=fig, t=title, e=exp, i=idx: self._save_trace_pdf(
                        f, t, e, i
                    ),
                ).grid(row=0, column=2, padx=2, sticky="ew")
                for col in range(3):
                    btn_row.columnconfigure(col, weight=1)
        self.card_widgets.append(card)
        return card

    def _apply_axis_limits(self, ax, ax2=None):
        """Apply user-specified limits: X to ax (twinx shares it), Y_l to ax, Y_r to ax2."""
        try:
            xmin_s = self._xlim_min_var.get().strip()
            xmax_s = self._xlim_max_var.get().strip()
            if xmin_s or xmax_s:
                cur = ax.get_xlim()
                ax.set_xlim(
                    float(xmin_s) if xmin_s else cur[0],
                    float(xmax_s) if xmax_s else cur[1],
                )
        except (ValueError, AttributeError):
            pass
        try:
            ymin_s = self._ylim_min_var.get().strip()
            ymax_s = self._ylim_max_var.get().strip()
            if ymin_s or ymax_s:
                cur = ax.get_ylim()
                ax.set_ylim(
                    float(ymin_s) if ymin_s else cur[0],
                    float(ymax_s) if ymax_s else cur[1],
                )
        except (ValueError, AttributeError):
            pass
        if ax2 is not None:
            try:
                ymin2_s = self._ylim2_min_var.get().strip()
                ymax2_s = self._ylim2_max_var.get().strip()
                if ymin2_s or ymax2_s:
                    cur2 = ax2.get_ylim()
                    ax2.set_ylim(
                        float(ymin2_s) if ymin2_s else cur2[0],
                        float(ymax2_s) if ymax2_s else cur2[1],
                    )
            except (ValueError, AttributeError):
                pass

    def _confirm_one(self, exp, idx, label, card):
        self.labeler.set_label(exp, idx, label)
        self._update_status()
        self._flash_card(card, "green")
        self._log(f"Confirmed {exp}:{idx} → {label}")

    def _reassign_one(self, exp, idx, label, card):
        self.labeler.set_label(exp, idx, label)
        self._update_status()
        self._flash_card(card, "orange")
        self._log(f"Assigned {exp}:{idx} → {label}")

    def _unlabel_one(self, exp, idx, card):
        self.labeler.remove_label(exp, idx)
        self._update_status()
        self._flash_card(card, "red")
        self._log(f"Unlabeled {exp}:{idx}")

    def _delete_label_class(self):
        target = self._get_review_label()
        if not target:
            messagebox.showinfo("Delete Class", "Select a label class first.")
            return
        affected = [k for k, v in self.labeler.labels.items() if v == target]
        if not messagebox.askyesno(
            "Delete Class",
            f"Remove '{target}' and unlabel {len(affected)} trace(s)?",
        ):
            return
        for key in affected:
            del self.labeler.labels[key]
        self.labeler.label_set.discard(target)
        self._update_status()
        self._log(f"Deleted class '{target}' ({len(affected)} traces unlabeled)")

    def _save_trace_pdf(self, fig, title, exp, idx):
        safe = (
            title.replace(" ", "_")
            .replace("#", "")
            .replace("=", "")
            .replace(".", "")
            .replace("/", "_")
            .replace("\\", "_")
            .strip("_")
        )
        out_dir = Path(self._labels_path).parent / "pdfs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{safe}.pdf"
        fig.savefig(path, bbox_inches="tight")
        self._log(f"Saved PDF: {path}")
        self._update_pdf_index(exp, idx, path.name)

    def _update_pdf_index(self, exp, idx, pdf_filename):
        import csv

        index_path = Path(self._labels_path).parent / "pdfs" / "index.csv"
        # Load existing entries
        entries = {}
        if index_path.exists():
            with open(index_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    row.setdefault("data_root", "")
                    key = (row["experiment"], int(row["id"]))
                    entries[key] = row
        # Get xy from track data (mean position over the track lifetime)
        tr = self.labeler.all_data[exp][idx]
        x_mean = float(np.mean(tr["x"]))
        y_mean = float(np.mean(tr["y"]))
        entries[(exp, idx)] = {
            "experiment": exp,
            "id": idx,
            "x": f"{x_mean:.3f}",
            "y": f"{y_mean:.3f}",
            "pdf": pdf_filename,
            "data_root": str(Path(config.DATA_ROOT).resolve()),
        }
        # Sort: group by experiment name, then ascending ID
        sorted_rows = sorted(
            entries.values(), key=lambda r: (r["experiment"], int(r["id"]))
        )
        with open(index_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["experiment", "id", "x", "y", "pdf", "data_root"]
            )
            writer.writeheader()
            writer.writerows(sorted_rows)
        self._log(f"Updated PDF index: {index_path}")

    def _flash_card(self, card, color):
        """Briefly flash the card border to confirm action."""
        try:
            orig_style = card.cget("style") or ""
        except Exception:
            orig_style = ""
        style_name = f"Flash.{color}.TLabelframe"
        style = ttk.Style()
        style.configure(style_name, bordercolor=color, relief="solid", borderwidth=2)
        card.configure(style=style_name)
        self.after(600, lambda: card.configure(style=orig_style))

    # ── Diagnostics ──────────────────────────────────────────────────────
    def _show_diagnostics(self):
        from sklearn.metrics import confusion_matrix as _cm

        if self.labeler.model is None:
            messagebox.showinfo("Diagnostics", "Train the model first.")
            return

        win = tk.Toplevel(self)
        win.title("Diagnostics")
        win.geometry("900x700")

        fig = Figure(figsize=(10, 8), dpi=100)

        # Confusion matrix
        ax1 = fig.add_subplot(221)
        labeled_mask = self.labeler.get_labeled_mask()
        label_arr = self.labeler.get_label_array()
        # HistGradientBoostingClassifier is trained on raw features — do not scale
        X_lab = self.labeler.feat_df.values[labeled_mask]
        y_true = label_arr[labeled_mask]
        y_pred = self.labeler.model.predict(X_lab)
        classes = self.labeler.classes
        cm = _cm(y_true, y_pred, labels=classes)
        ax1.imshow(cm, cmap="Blues")
        ax1.set_xticks(range(len(classes)))
        ax1.set_yticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax1.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax1.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_title("Confusion Matrix")

        # Feature importance
        ax2 = fig.add_subplot(222)
        if hasattr(self.labeler.model, "feature_importances_"):
            imp = self.labeler.model.feature_importances_
        else:
            from sklearn.inspection import permutation_importance

            r = permutation_importance(
                self.labeler.model,
                X_lab,
                y_true,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
            imp = r.importances_mean
        names = self.labeler.feat_df.columns
        top_n = min(15, len(names))
        order = np.argsort(imp)[::-1][:top_n]
        ax2.barh(range(top_n), imp[order[::-1]], color="steelblue")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels([names[i] for i in order[::-1]], fontsize=7)
        ax2.set_xlabel("Importance")
        ax2.set_title(f"Top {top_n} Features")

        # Training history
        if len(self.labeler.history) > 1:
            ax3 = fig.add_subplot(223)
            rounds = [h["round"] for h in self.labeler.history]
            train_acc = [h["train_acc"] for h in self.labeler.history]
            cv_acc = [h["cv_acc"] for h in self.labeler.history]
            ax3.plot(rounds, train_acc, "o-", label="Train")
            ax3.plot(rounds, cv_acc, "s--", label="CV")
            ax3.set_xlabel("Round")
            ax3.set_ylabel("Accuracy")
            ax3.legend()
            ax3.set_title("Training History")

            ax4 = fig.add_subplot(224)
            n_lab = [h["n_labeled"] for h in self.labeler.history]
            ax4.bar(rounds, n_lab, color="gray", alpha=0.6)
            ax4.set_xlabel("Round")
            ax4.set_ylabel("# Labeled")
            ax4.set_title("Labels Over Rounds")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def _save_predictions(self):
        """Save model probabilities for all traces to a CSV file."""
        import csv

        if self.labeler.proba is None or self.labeler.classes is None:
            messagebox.showinfo("Save Predictions", "Train the model first.")
            return
        out_dir = Path(self._labels_path).parent
        path = filedialog.asksaveasfilename(
            title="Save predictions CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="predictions.csv",
            initialdir=str(out_dir),
        )
        if not path:
            return
        classes = list(self.labeler.classes)
        predicted = self.labeler.classes[np.argmax(self.labeler.proba, axis=1)]
        fieldnames = [
            "experiment",
            "id",
            "predicted_label",
            "manual_label",
            "is_ignored",
        ] + [f"prob_{c}" for c in classes]
        rows = []
        for gi, row in self.labeler.index_df.iterrows():
            exp, idx = row.exp, int(row.idx)
            probs = self.labeler.proba[gi]
            r = {
                "experiment": exp,
                "id": idx,
                "predicted_label": predicted[gi],
                "manual_label": self.labeler.get_label(exp, idx),
                "is_ignored": self.labeler.get_sub_label(exp, idx) == IGNORED,
            }
            for c, p in zip(classes, probs):
                r[f"prob_{c}"] = f"{p:.6f}"
            rows.append(r)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self._log(f"Saved predictions ({len(rows)} traces) → {path}")

    # ── Save / Load ──────────────────────────────────────────────────────
    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=os.path.basename(self.labeler.save_path),
            initialdir=os.path.dirname(self.labeler.save_path) or ".",
        )
        if not path:
            return
        self.labeler.save(path)
        self._log(f"Saved {self.labeler.n_labeled} labels → {path}")

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")],
            initialdir=os.path.dirname(self.labeler.save_path) or ".",
        )
        if not path:
            return
        n = self.labeler.load(path)
        if self.labeler.n_ignored > 0:
            self._filter_status_var.set(
                f"{self.labeler.n_ignored}/{self.labeler.N} traces ignored"
            )
            # Auto-enable "Show ignored traces" so filter labels appear in the combo
            self._show_ignored_var.set(True)
        else:
            self._filter_status_var.set("No filters active")
            self._show_ignored_var.set(False)
        self._update_status()
        self._log(f"Loaded {n} labels from {path}")

    def _on_close(self):
        if self.labeler is not None and self.labeler.n_labeled > 0:
            if messagebox.askyesno("Quit", "Save labels before closing?"):
                self._save()
        self.destroy()


def LabelerApp(pkl_path=None, labels_path=None, feature_mode="pelt", cnn_device="cpu"):
    """Factory: creates a tk.Tk subclass with all LabelerApp methods."""

    class _App(_LabelerAppMethods, tk.Tk):
        pass

    return _App(
        pkl_path=pkl_path,
        labels_path=labels_path,
        feature_mode=feature_mode,
        cnn_device=cnn_device,
    )
