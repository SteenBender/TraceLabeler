"""Filter configuration dialog for marking traces as valid/ignored."""

import tkinter as tk
from tkinter import ttk, messagebox


class FilterDialog(tk.Toplevel):
    """Modal dialog for configuring and applying trace filters."""

    def __init__(self, parent, channels, current_config=None):
        super().__init__(parent)
        self.title("Trace Filters")
        self.geometry("520x620")
        self.resizable(False, True)
        self.grab_set()

        self._channels = channels
        self._result = None  # will be set to the config dict on Apply

        # Pre-fill from current config if re-opening
        cfg = current_config or {}

        # ── Buttons (pack FIRST at bottom so they're always visible) ──
        btn_frame = ttk.Frame(self, padding=8)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 4))
        ttk.Button(btn_frame, text="Apply Filters", command=self._apply).pack(
            side=tk.LEFT, padx=4, ipadx=10
        )
        ttk.Button(btn_frame, text="Clear All Filters", command=self._clear).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=4
        )

        # ── Scrollable area for filter sections ──
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mousewheel scrolling
        def _on_mousewheel(event):
            # macOS sends delta in units, Linux/Windows in multiples of 120
            if event.delta:
                canvas.yview_scroll(-event.delta, "units")
            elif event.num == 4:
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                canvas.yview_scroll(3, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # macOS / Windows
        canvas.bind_all("<Button-4>", _on_mousewheel)  # Linux scroll up
        canvas.bind_all("<Button-5>", _on_mousewheel)  # Linux scroll down

        # Unbind when dialog closes so we don't leak global bindings
        def _on_destroy(event):
            if event.widget is self:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")

        self.bind("<Destroy>", _on_destroy)

        # ── Existence & Recruitment filter ──
        ex_cfg = cfg.get("existence", {})
        ex_frame = ttk.LabelFrame(scroll_frame, text="Existence & Recruitment Check", padding=8)
        ex_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        self._ex_enabled = tk.BooleanVar(value=ex_cfg.get("enabled", False))
        ttk.Checkbutton(ex_frame, text="Enable", variable=self._ex_enabled).pack(
            anchor=tk.W
        )

        ttk.Label(
            ex_frame,
            text="Minimum steps per channel (0 = no requirement):",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(4, 2))

        self._ex_step_vars = {}
        self._ex_dir_vars = {}
        ex_min_steps = ex_cfg.get("min_steps", {})
        ex_step_direction = ex_cfg.get("step_direction", {})
        ex_grid = ttk.Frame(ex_frame)
        ex_grid.pack(fill=tk.X)
        ttk.Label(ex_grid, text="Channel", width=14).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(ex_grid, text="Min steps", width=8).grid(row=0, column=1)
        ttk.Label(ex_grid, text="Direction", width=10).grid(row=0, column=2)
        for i, ch in enumerate(channels):
            ttk.Label(ex_grid, text=ch, width=14).grid(row=i + 1, column=0, sticky=tk.W)
            var = tk.StringVar(value=str(ex_min_steps.get(ch, 0)))
            self._ex_step_vars[ch] = var
            ttk.Entry(ex_grid, textvariable=var, width=8).grid(
                row=i + 1, column=1, padx=4, pady=1
            )
            dir_var = tk.StringVar(value=ex_step_direction.get(ch, "positive"))
            self._ex_dir_vars[ch] = dir_var
            ttk.Combobox(
                ex_grid,
                textvariable=dir_var,
                values=["positive", "negative", "both"],
                state="readonly",
                width=9,
            ).grid(row=i + 1, column=2, padx=4, pady=1)

        # ── Intensity Stability filter ──
        in_cfg = cfg.get("intensity", {})
        in_frame = ttk.LabelFrame(scroll_frame, text="Intensity Stability", padding=8)
        in_frame.pack(fill=tk.X, padx=8, pady=4)

        self._in_enabled = tk.BooleanVar(value=in_cfg.get("enabled", False))
        ttk.Checkbutton(in_frame, text="Enable", variable=self._in_enabled).pack(
            anchor=tk.W
        )

        ttk.Label(
            in_frame,
            text="Max absolute step size per channel (0 = no limit):",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(4, 2))

        self._in_max_vars = {}
        in_max = in_cfg.get("max_step_size", {})
        in_grid = ttk.Frame(in_frame)
        in_grid.pack(fill=tk.X)
        for i, ch in enumerate(channels):
            ttk.Label(in_grid, text=ch, width=14).grid(row=i, column=0, sticky=tk.W)
            var = tk.StringVar(value=str(in_max.get(ch, 0)))
            self._in_max_vars[ch] = var
            ttk.Entry(in_grid, textvariable=var, width=8).grid(
                row=i, column=1, padx=4, pady=1
            )

        # ── Temporal Order filter ──
        te_cfg = cfg.get("temporal", {})
        te_frame = ttk.LabelFrame(scroll_frame, text="Temporal Order", padding=8)
        te_frame.pack(fill=tk.X, padx=8, pady=4)

        self._te_enabled = tk.BooleanVar(value=te_cfg.get("enabled", False))
        ttk.Checkbutton(te_frame, text="Enable", variable=self._te_enabled).pack(
            anchor=tk.W
        )

        ttk.Label(
            te_frame,
            text="Required order of first non-noise step (drag or number):",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(4, 2))

        # Order: show channels with rank spinboxes
        existing_order = te_cfg.get("channel_order", channels[:])
        self._te_order_vars = {}
        te_order_grid = ttk.Frame(te_frame)
        te_order_grid.pack(fill=tk.X)
        ttk.Label(te_order_grid, text="Channel", width=14).grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Label(te_order_grid, text="Rank", width=6).grid(row=0, column=1)
        ttk.Label(te_order_grid, text="Noise thresh", width=10).grid(row=0, column=2)

        te_noise = te_cfg.get("noise_thresh", {})
        self._te_noise_vars = {}
        for i, ch in enumerate(channels):
            ttk.Label(te_order_grid, text=ch, width=14).grid(
                row=i + 1, column=0, sticky=tk.W
            )
            # Rank: position in the required order (1-based), 0 = skip
            try:
                rank = existing_order.index(ch) + 1
            except ValueError:
                rank = 0
            rank_var = tk.StringVar(value=str(rank))
            self._te_order_vars[ch] = rank_var
            ttk.Spinbox(
                te_order_grid,
                textvariable=rank_var,
                from_=0,
                to=len(channels),
                width=4,
            ).grid(row=i + 1, column=1, padx=4, pady=1)

            noise_var = tk.StringVar(value=str(te_noise.get(ch, 0.0)))
            self._te_noise_vars[ch] = noise_var
            ttk.Entry(te_order_grid, textvariable=noise_var, width=8).grid(
                row=i + 1, column=2, padx=4, pady=1
            )

        # Leniency
        len_row = ttk.Frame(te_frame)
        len_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(len_row, text="Leniency (frames):").pack(side=tk.LEFT)
        self._te_leniency_var = tk.StringVar(value=str(te_cfg.get("leniency", 0.0)))
        ttk.Entry(len_row, textvariable=self._te_leniency_var, width=8).pack(
            side=tk.LEFT, padx=4
        )

    def _build_config(self):
        """Read all widgets and return a filter_config dict."""
        config = {}

        # Existence
        min_steps = {}
        for ch, var in self._ex_step_vars.items():
            try:
                val = int(var.get())
            except ValueError:
                val = 0
            if val > 0:
                min_steps[ch] = val
        step_direction = {
            ch: var.get() for ch, var in self._ex_dir_vars.items()
        }
        config["existence"] = {
            "enabled": self._ex_enabled.get(),
            "channels": [ch for ch in self._channels if ch in min_steps],
            "min_steps": min_steps,
            "step_direction": step_direction,
        }

        # Intensity
        max_step_size = {}
        for ch, var in self._in_max_vars.items():
            try:
                val = float(var.get())
            except ValueError:
                val = 0.0
            if val > 0:
                max_step_size[ch] = val
        config["intensity"] = {
            "enabled": self._in_enabled.get(),
            "channels": [ch for ch in self._channels if ch in max_step_size],
            "max_step_size": max_step_size,
        }

        # Temporal order
        order_ranks = {}
        for ch, var in self._te_order_vars.items():
            try:
                rank = int(var.get())
            except ValueError:
                rank = 0
            if rank > 0:
                order_ranks[ch] = rank
        # Sort by rank to get the channel order
        channel_order = [
            ch for ch, _ in sorted(order_ranks.items(), key=lambda x: x[1])
        ]
        noise_thresh = {}
        for ch, var in self._te_noise_vars.items():
            try:
                val = float(var.get())
            except ValueError:
                val = 0.0
            noise_thresh[ch] = val
        try:
            leniency = float(self._te_leniency_var.get())
        except ValueError:
            leniency = 0.0
        config["temporal"] = {
            "enabled": self._te_enabled.get(),
            "channel_order": channel_order,
            "noise_thresh": noise_thresh,
            "leniency": leniency,
        }

        return config

    def _apply(self):
        self._result = self._build_config()
        self.destroy()

    def _clear(self):
        self._result = "clear"
        self.destroy()

    def get_result(self):
        """Call after wait_window(). Returns config dict, 'clear', or None."""
        return self._result
