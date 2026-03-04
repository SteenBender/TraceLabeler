"""Animated TIFF patch viewer — secondary Toplevel window."""

import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from .. import config
from ..labeler import UNLABELED
from ..patch_loader import _patch_cache, _load_patches_for_page


class _StopToken:
    """Tiny mutable flag used to cancel a running animation."""

    __slots__ = ("stopped",)

    def __init__(self):
        self.stopped = False


class PatchViewerWindow:
    """Secondary window: shows animated TIFF crops for a list of trace candidates."""

    HALF = 5        # crop half-size → 10×10 px crops
    N_PER_PAGE = 6
    FPS = 10        # playback frames per second

    def __init__(self, parent_app, candidates, start_page=0):
        self._app = parent_app
        self._labeler = parent_app.labeler
        self.candidates = list(candidates)
        self.page = start_page
        self._figures = []
        self._stop_tokens = []   # one per card; set .stopped=True in _clear()
        self._loading = False

        self.win = tk.Toplevel(parent_app)
        self.win.title("Patch Viewer")
        self.win.geometry("1000x780")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._load_page()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.status_var = tk.StringVar(value="Initialising …")
        ttk.Label(
            self.win,
            textvariable=self.status_var,
            relief=tk.GROOVE,
            padding=4,
        ).pack(fill=tk.X, padx=4, pady=(4, 0))

        nav = ttk.Frame(self.win)
        nav.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(nav, text="◀ Prev", command=self._prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Next ▶", command=self._next).pack(side=tk.LEFT, padx=2)

        self.content = ttk.Frame(self.win)
        self.content.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Pagination ───────────────────────────────────────────────────────────
    def _page_cands(self):
        s = self.page * self.N_PER_PAGE
        return self.candidates[s : s + self.N_PER_PAGE]

    def _prev(self):
        if self.page > 0 and not self._loading:
            self.page -= 1
            self._load_page()

    def _next(self):
        max_page = max(0, (len(self.candidates) - 1) // self.N_PER_PAGE)
        if self.page < max_page and not self._loading:
            self.page += 1
            self._load_page()

    # ── TIFF loading (background thread) ────────────────────────────────────
    def _load_page(self):
        self._clear()
        page_cands = self._page_cands()
        needs_load = any(
            (c["exp"], ch, c["idx"]) not in _patch_cache
            for c in page_cands
            for ch in config.CHANNELS
        )
        if not needs_load:
            self._render()
            return

        self._loading = True
        exps = list(dict.fromkeys(c["exp"] for c in page_cands))
        self.status_var.set(f"Loading frames for {', '.join(exps)} …")

        def worker():
            _load_patches_for_page(page_cands, self._labeler, half=self.HALF)
            self.win.after(0, self._on_loaded)

        threading.Thread(target=worker, daemon=True).start()

    def _on_loaded(self):
        self._loading = False
        self._render()

    def _evict_distant_pages(self):
        """Remove cached patches for pages more than 1 away from the current page."""
        keep_pages = {self.page - 1, self.page, self.page + 1}
        for p_idx, cand in enumerate(self.candidates):
            p = p_idx // self.N_PER_PAGE
            if p not in keep_pages:
                for ch in config.CHANNELS:
                    _patch_cache.pop((cand["exp"], ch, cand["idx"]), None)

    def _prefetch_adjacent(self):
        """Silently load patches for the next (and previous) page in the background."""
        max_page = max(0, (len(self.candidates) - 1) // self.N_PER_PAGE)
        pages_to_fetch = [
            p for p in (self.page + 1, self.page - 1) if 0 <= p <= max_page
        ]
        if not pages_to_fetch:
            return
        cands = []
        for p in pages_to_fetch:
            s = p * self.N_PER_PAGE
            cands.extend(self.candidates[s : s + self.N_PER_PAGE])
        # Only bother if anything is actually missing from cache
        if not any(
            (c["exp"], ch, c["idx"]) not in _patch_cache
            for c in cands
            for ch in config.CHANNELS
        ):
            return
        labeler = self._labeler
        half = self.HALF
        threading.Thread(
            target=lambda: _load_patches_for_page(cands, labeler, half=half),
            daemon=True,
        ).start()

    # ── Rendering ────────────────────────────────────────────────────────────
    def _clear(self):
        # Stop all running animations before destroying widgets
        for tok in self._stop_tokens:
            tok.stopped = True
        self._stop_tokens.clear()
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
        for w in self.content.winfo_children():
            w.destroy()

    def _render(self):
        self._clear()
        page_cands = self._page_cands()
        n_total = len(self.candidates)
        max_page = max(0, (n_total - 1) // self.N_PER_PAGE)
        self.status_var.set(
            f"Page {self.page + 1}/{max_page + 1}  —  {n_total} traces"
        )
        grid = ttk.Frame(self.content)
        grid.pack(fill=tk.BOTH, expand=True)
        ncols = 2
        for i, cand in enumerate(page_cands):
            r, c = divmod(i, ncols)
            card = self._build_card(grid, cand)
            card.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            grid.columnconfigure(c, weight=1)

        self._evict_distant_pages()
        self._prefetch_adjacent()

    def _build_card(self, parent, cand):
        exp, idx = cand["exp"], cand["idx"]
        label = self._labeler.get_label(exp, idx)
        label_str = label if label != UNLABELED else "unlabeled"
        frame_arr = np.asarray(self._labeler.all_data[exp][idx]["frames"], dtype=int)
        n_frames = len(frame_arr)

        card = ttk.LabelFrame(
            parent,
            text=f"{exp} #{idx}  [{label_str}]  {n_frames} frames",
            padding=4,
        )

        # ── Matplotlib figure with one subplot per channel ──────────────────
        fig = Figure(figsize=(5.5, 2.0), dpi=90)
        ims = []
        ch_ranges = []  # (vmin, vmax) per channel for stable display across frames
        blank = np.zeros((2 * self.HALF, 2 * self.HALF), dtype=np.uint16)

        for ci, ch in enumerate(config.CHANNELS):
            ax = fig.add_subplot(1, len(config.CHANNELS), ci + 1)
            patches = _patch_cache.get((exp, ch, idx))
            if patches is not None and patches.shape[0] > 0:
                vmin, vmax = int(patches.min()), int(patches.max())
                im = ax.imshow(
                    patches[0],
                    cmap="gray",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                vmin, vmax = 0, 1
                im = ax.imshow(blank, cmap="gray", vmin=0, vmax=1)
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#888",
                )
                ax.set_facecolor("#111111")
            ax.set_title(ch, fontsize=7, color=self._labeler.channel_colors[ch])
            ax.axis("off")
            ims.append(im)
            ch_ranges.append((vmin, vmax))

        fig.tight_layout(pad=0.3)
        self._figures.append(fig)
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Playback controls ────────────────────────────────────────────────
        stop_tok = _StopToken()
        self._stop_tokens.append(stop_tok)

        ctrl = ttk.Frame(card)
        ctrl.pack(fill=tk.X, pady=(2, 0))

        frame_var = tk.IntVar(value=0)
        playing = [False]
        job_id = [None]

        def show_frame(fi):
            for ci, ch in enumerate(config.CHANNELS):
                patches = _patch_cache.get((exp, ch, idx))
                if patches is None or patches.shape[0] == 0:
                    continue
                fi_clip = min(fi, patches.shape[0] - 1)
                ims[ci].set_data(patches[fi_clip])
            canvas.draw_idle()
            lbl_var.set(f"{fi + 1}/{n_frames}")

        def step():
            if stop_tok.stopped or not playing[0]:
                return
            fi = (frame_var.get() + 1) % n_frames
            frame_var.set(fi)
            slider.set(fi)
            show_frame(fi)
            job_id[0] = self.win.after(1000 // self.FPS, step)

        def on_slider(val):
            fi = int(float(val))
            frame_var.set(fi)
            show_frame(fi)

        def toggle_play():
            if stop_tok.stopped:
                return
            playing[0] = not playing[0]
            play_btn.configure(text="⏸" if playing[0] else "▶")
            if playing[0]:
                step()
            elif job_id[0] is not None:
                self.win.after_cancel(job_id[0])
                job_id[0] = None

        play_btn = ttk.Button(ctrl, text="▶", width=3, command=toggle_play)
        play_btn.pack(side=tk.LEFT, padx=(0, 4))

        slider = ttk.Scale(
            ctrl,
            from_=0,
            to=max(0, n_frames - 1),
            orient=tk.HORIZONTAL,
            variable=frame_var,
            command=on_slider,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lbl_var = tk.StringVar(value=f"1/{n_frames}")
        ttk.Label(ctrl, textvariable=lbl_var, width=8).pack(side=tk.LEFT, padx=(4, 0))

        # Start playback automatically
        toggle_play()

        return card

    def _on_close(self):
        self._clear()
        self.win.destroy()
