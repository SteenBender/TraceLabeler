"""UMAP Explorer window — interactive 2-D projection for trace labeling."""

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors as mcolors  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import numpy as np

from .. import config  # noqa: F401
from ..labeler import UNLABELED


class UmapViewerWindow:
    """Toplevel window that computes a UMAP projection and supports lasso labeling."""

    def __init__(self, parent_app):
        self._app = parent_app
        self._labeler = parent_app.labeler

        self._embedding = None       # (M, 2)
        self._point_gi = None        # (M,) global indices kept in the embedding
        self._selected_mask = None   # (M,) bool
        self._computing = False
        self._hover_nearest = ()   # tuple of global indices currently shown
        self._hover_debounce = None
        self._lasso = None

        self.win = tk.Toplevel(parent_app)
        self.win.title("UMAP Explorer")
        self.win.geometry("1420x840")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Left settings panel ──────────────────────────────────────────
        left = ttk.Frame(self.win, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)

        ttk.Label(left, text="UMAP Settings", font=("Helvetica", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )

        ttk.Label(left, text="Features:").pack(anchor=tk.W)
        self._feat_var = tk.StringVar(
            value="Pretrained" if self._labeler.feature_mode == "pretrained" else "PELT"
        )
        ttk.Combobox(
            left,
            textvariable=self._feat_var,
            values=["PELT", "Pretrained"],
            state="readonly",
            width=14,
        ).pack(fill=tk.X, pady=(0, 6))

        self._include_ignored_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="Include ignored traces", variable=self._include_ignored_var
        ).pack(anchor=tk.W, pady=(0, 6))

        ttk.Label(left, text="n_neighbors:").pack(anchor=tk.W)
        self._n_neighbors_var = tk.IntVar(value=15)
        nn_row = ttk.Frame(left)
        nn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Scale(
            nn_row,
            from_=2,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self._n_neighbors_var,
            length=130,
        ).pack(side=tk.LEFT)
        self._nn_lbl = ttk.Label(nn_row, text="15", width=4)
        self._nn_lbl.pack(side=tk.LEFT)
        self._n_neighbors_var.trace_add(
            "write",
            lambda *_: self._nn_lbl.config(text=str(self._n_neighbors_var.get())),
        )

        ttk.Label(left, text="min_dist:").pack(anchor=tk.W)
        self._min_dist_var = tk.StringVar(value="0.1")
        ttk.Entry(left, textvariable=self._min_dist_var, width=8).pack(
            anchor=tk.W, pady=(0, 6)
        )

        ttk.Label(left, text="Metric:").pack(anchor=tk.W)
        self._metric_var = tk.StringVar(value="euclidean")
        ttk.Combobox(
            left,
            textvariable=self._metric_var,
            values=["euclidean", "cosine", "manhattan", "correlation"],
            state="readonly",
            width=14,
        ).pack(fill=tk.X, pady=(0, 8))

        self._compute_btn = ttk.Button(
            left, text="Compute UMAP", command=self._compute_umap
        )
        self._compute_btn.pack(fill=tk.X, pady=4)

        self._status_var = tk.StringVar(value="Adjust settings and click Compute.")
        ttk.Label(
            left,
            textvariable=self._status_var,
            wraplength=190,
            foreground="gray",
            font=("Helvetica", 8),
        ).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── Selection & labeling ──────────────────────────────────────────
        ttk.Label(
            left, text="Selection & Labeling", font=("Helvetica", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 4))

        self._sel_count_var = tk.StringVar(value="No selection")
        ttk.Label(
            left,
            textvariable=self._sel_count_var,
            foreground="cyan",
            font=("Helvetica", 8),
        ).pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(left, text="New label:").pack(anchor=tk.W)
        self._new_label_var = tk.StringVar()
        new_label_entry = ttk.Entry(left, textvariable=self._new_label_var, width=16)
        new_label_entry.pack(fill=tk.X, pady=(0, 2))
        new_label_entry.bind("<Return>", lambda _e: self._add_new_label())
        ttk.Button(left, text="Add Label", command=self._add_new_label).pack(
            fill=tk.X, pady=(0, 6)
        )

        ttk.Label(left, text="Assign label:").pack(anchor=tk.W)
        self._assign_label_var = tk.StringVar()
        self._assign_combo = ttk.Combobox(
            left, textvariable=self._assign_label_var, state="readonly", width=14
        )
        self._assign_combo.pack(fill=tk.X, pady=(0, 4))

        ttk.Button(left, text="Label Selected", command=self._label_selected).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(left, text="Clear Selection", command=self._clear_selection).pack(
            fill=tk.X, pady=2
        )
        ttk.Label(
            left,
            text="Draw a lasso on the plot\nto select a cluster region.",
            foreground="gray",
            font=("Helvetica", 8),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        # ── Center: scatter plot ──────────────────────────────────────────
        center = ttk.Frame(self.win)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._scatter_fig = Figure(figsize=(7, 6), dpi=100)
        self._scatter_ax = self._scatter_fig.add_subplot(111)
        self._scatter_ax.set_facecolor("#1e1e1e")
        self._scatter_fig.patch.set_facecolor("#1e1e1e")

        self._scatter_canvas = FigureCanvasTkAgg(self._scatter_fig, master=center)
        self._scatter_canvas.draw()
        self._scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._scatter_canvas.mpl_connect("motion_notify_event", self._on_hover)

        # ── Right: hover trace preview ────────────────────────────────────
        right = ttk.LabelFrame(self.win, text="Hover Preview", padding=4)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)

        self._preview_fig = Figure(figsize=(3.8, 7.5), dpi=90)
        self._preview_fig.patch.set_facecolor("#1e1e1e")

        self._preview_canvas = FigureCanvasTkAgg(self._preview_fig, master=right)
        self._preview_canvas.draw()
        self._preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._hover_info_var = tk.StringVar(value="Hover over a point to preview")
        ttk.Label(
            right,
            textvariable=self._hover_info_var,
            font=("Helvetica", 8),
            wraplength=330,
        ).pack(anchor=tk.W, pady=(4, 0))

    # ── UMAP computation ──────────────────────────────────────────────────

    def _compute_umap(self):
        if self._computing:
            return

        labeler = self._labeler
        feat_mode = self._feat_var.get()

        if feat_mode == "Pretrained":
            if labeler._embed_features is None:
                messagebox.showwarning(
                    "UMAP",
                    "Pretrained embeddings not available.\n"
                    "Run 'Pretrain Encoder' first.",
                )
                return
            X_all = labeler._embed_features
        else:
            X_all = labeler.feat_df.values

        include_ignored = self._include_ignored_var.get()
        keep = (
            np.ones(labeler.N, dtype=bool)
            if include_ignored
            else ~labeler.get_ignored_mask()
        )
        point_gi = np.where(keep)[0]
        X = X_all[keep].astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            n_neighbors = max(2, int(self._n_neighbors_var.get()))
            min_dist = float(self._min_dist_var.get())
        except ValueError:
            messagebox.showerror("UMAP", "Invalid n_neighbors or min_dist value.")
            return

        metric = self._metric_var.get()
        self._computing = True
        self._compute_btn.config(state=tk.DISABLED)
        self._status_var.set(f"Computing UMAP on {len(X)} traces…")
        self.win.update_idletasks()

        def _worker():
            try:
                import umap as umap_lib  # noqa: PLC0415

                reducer = umap_lib.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    n_components=2,
                    random_state=42,
                )
                emb = reducer.fit_transform(X)
                self.win.after(0, lambda: self._on_umap_done(emb, point_gi))
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.win.after(0, lambda: self._on_umap_error(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_umap_error(self, msg):
        self._computing = False
        self._compute_btn.config(state=tk.NORMAL)
        self._status_var.set(f"Error: {msg}")
        messagebox.showerror("UMAP Error", msg)

    def _on_umap_done(self, embedding, point_gi):
        self._computing = False
        self._compute_btn.config(state=tk.NORMAL)
        self._embedding = embedding
        self._point_gi = point_gi
        self._selected_mask = np.zeros(len(embedding), dtype=bool)
        self._status_var.set(f"Done — {len(embedding)} points.")
        self._update_assign_combo()
        self._draw_scatter()
        self._reattach_lasso()

    # ── Drawing ───────────────────────────────────────────────────────────

    def _build_color_arrays(self):
        """Return (N_pts, 4) RGBA array and a {label: color} dict for the legend."""
        labeler = self._labeler
        label_arr = labeler.get_label_array()
        known = sorted(labeler.label_set) if labeler.label_set else []
        cmap = plt.get_cmap("tab20", max(len(known), 1))
        lbl2color = {lbl: cmap(i) for i, lbl in enumerate(known)}
        gray = np.array([0.45, 0.45, 0.45, 0.45])
        colors = []
        for gi in self._point_gi:
            lbl = label_arr[gi]
            if lbl == UNLABELED:
                colors.append(gray)
            else:
                c = lbl2color.get(lbl, gray)
                colors.append(np.array(c))
        return np.array(colors), lbl2color

    def _draw_scatter(self, highlight_sel=False):
        ax = self._scatter_ax
        ax.cla()
        ax.set_facecolor("#1e1e1e")
        self._scatter_fig.patch.set_facecolor("#1e1e1e")

        if self._embedding is None:
            self._scatter_canvas.draw()
            return

        emb = self._embedding
        colors, lbl2color = self._build_color_arrays()

        if highlight_sel and self._selected_mask is not None and self._selected_mask.any():
            unsel = ~self._selected_mask
            if unsel.any():
                ax.scatter(
                    emb[unsel, 0], emb[unsel, 1],
                    c=colors[unsel], s=7, linewidths=0, alpha=0.3, rasterized=True,
                )
            sel_idx = np.where(self._selected_mask)[0]
            ax.scatter(
                emb[sel_idx, 0], emb[sel_idx, 1],
                c=colors[sel_idx], s=20, linewidths=0.7,
                edgecolors="white", alpha=1.0, rasterized=True,
            )
        else:
            ax.scatter(
                emb[:, 0], emb[:, 1], c=colors, s=7,
                linewidths=0, alpha=0.7, rasterized=True,
            )

        for lbl, color in lbl2color.items():
            ax.scatter([], [], c=[color], s=20, label=lbl)
        ax.scatter([], [], c=[(0.45, 0.45, 0.45, 0.8)], s=20, label=UNLABELED)
        ax.legend(
            loc="upper right", fontsize=7, framealpha=0.3,
            labelcolor="white", markerscale=1.5,
        )
        ax.set_xlabel("UMAP 1", color="#cccccc", fontsize=9)
        ax.set_ylabel("UMAP 2", color="#cccccc", fontsize=9)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

        self._scatter_fig.tight_layout(pad=0.5)
        self._scatter_canvas.draw()

    # ── Lasso ─────────────────────────────────────────────────────────────

    def _reattach_lasso(self):
        if self._lasso is not None:
            try:
                self._lasso.disconnect_events()
            except Exception:
                pass
        self._lasso = LassoSelector(
            self._scatter_ax, self._on_lasso_select, useblit=False
        )

    def _on_lasso_select(self, verts):
        if self._embedding is None:
            return
        path = Path(verts)
        self._selected_mask = path.contains_points(self._embedding)
        n = int(self._selected_mask.sum())
        self._sel_count_var.set(f"{n} point{'s' if n != 1 else ''} selected")
        self._draw_scatter(highlight_sel=True)
        self._reattach_lasso()

    def _clear_selection(self):
        if self._embedding is None:
            return
        self._selected_mask = np.zeros(len(self._embedding), dtype=bool)
        self._sel_count_var.set("No selection")
        self._draw_scatter()
        self._reattach_lasso()

    # ── Labeling ──────────────────────────────────────────────────────────

    def _add_new_label(self):
        name = self._new_label_var.get().strip()
        if not name:
            return
        self._labeler.label_set.add(name)
        self._new_label_var.set("")
        self._update_assign_combo()
        self._assign_label_var.set(name)
        try:
            self._app._update_status()
            self._app._log(f"Added label: '{name}'")
        except Exception:
            pass

    def _update_assign_combo(self):
        labels = sorted(self._labeler.label_set) if self._labeler.label_set else []
        self._assign_combo["values"] = labels
        if labels and self._assign_label_var.get() not in labels:
            self._assign_label_var.set(labels[0])

    def _label_selected(self):
        if self._embedding is None or self._selected_mask is None or not self._selected_mask.any():
            messagebox.showinfo(
                "Label Selected",
                "Draw a lasso on the scatter plot to select traces first.",
            )
            return
        lbl = self._assign_label_var.get()
        if not lbl:
            messagebox.showwarning("Label Selected", "Choose a label from the dropdown.")
            return

        labeler = self._labeler
        indices = self._point_gi[self._selected_mask]
        n = 0
        for gi in indices:
            row = labeler.index_df.iloc[gi]
            labeler.set_label(row.exp, int(row.idx), lbl)
            n += 1

        self._draw_scatter(highlight_sel=True)
        try:
            self._app._update_status()
            self._app._log(f"UMAP: labeled {n} trace{'s' if n != 1 else ''} as '{lbl}'")
        except Exception:
            pass
        messagebox.showinfo(
            "Labeled", f"Labeled {n} trace{'s' if n != 1 else ''} as '{lbl}'."
        )

    # ── Hover preview ─────────────────────────────────────────────────────

    def _on_hover(self, event):
        if event.inaxes is not self._scatter_ax or self._embedding is None:
            return
        if self._hover_debounce is not None:
            self.win.after_cancel(self._hover_debounce)
        xd, yd = event.xdata, event.ydata
        self._hover_debounce = self.win.after(
            60, lambda: self._do_hover(xd, yd)
        )

    _N_HOVER = 3  # number of nearest traces shown in the preview panel

    def _do_hover(self, x, y):
        if x is None or y is None or self._embedding is None:
            return
        emb = self._embedding
        x_span = (emb[:, 0].max() - emb[:, 0].min()) or 1.0
        y_span = (emb[:, 1].max() - emb[:, 1].min()) or 1.0
        dists = np.hypot(
            (emb[:, 0] - x) / x_span,
            (emb[:, 1] - y) / y_span,
        )
        order = np.argsort(dists)
        if dists[order[0]] > 0.12:
            return

        n = min(self._N_HOVER, len(order))
        nearest_gis = tuple(int(self._point_gi[order[i]]) for i in range(n))
        if nearest_gis == self._hover_nearest:
            return
        self._hover_nearest = nearest_gis

        labeler = self._labeler
        info_parts = []
        for gi in nearest_gis:
            row = labeler.index_df.iloc[gi]
            lbl = labeler.get_label(row.exp, int(row.idx))
            lbl_str = lbl if lbl != UNLABELED else "unlabeled"
            info_parts.append(f"{row.exp} #{int(row.idx)} [{lbl_str}]")
        self._hover_info_var.set("  |  ".join(info_parts))

        # Rebuild figure with one subplot per trace to avoid twinx accumulation
        self._preview_fig.clf()
        for i, gi in enumerate(nearest_gis):
            row = labeler.index_df.iloc[gi]
            exp, idx = row.exp, int(row.idx)
            ax = self._preview_fig.add_subplot(n, 1, i + 1)
            ax.set_facecolor("#1e1e1e")
            labeler.plot_trace(exp, idx, ax=ax)
        self._preview_fig.tight_layout(pad=0.4, h_pad=0.8)
        self._preview_canvas.draw()

    # ── Cleanup ───────────────────────────────────────────────────────────

    def _on_close(self):
        if self._lasso is not None:
            try:
                self._lasso.disconnect_events()
            except Exception:
                pass
        plt.close(self._scatter_fig)
        plt.close(self._preview_fig)
        self.win.destroy()
