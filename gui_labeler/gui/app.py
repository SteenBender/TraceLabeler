"""Main tkinter application window for the active-learning trace labeler."""

import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from ..labeler import ActiveLabeler, UNLABELED
from .patch_viewer import PatchViewerWindow
from .pelt_tuner import PeltTunerWindow


class _LabelerAppMethods:
    """Main application window (mixed into tk.Tk subclass by LabelerApp factory)."""

    def __init__(self, labeler: ActiveLabeler):
        tk.Tk.__init__(self)
        self.labeler = labeler
        self.title("Active Trace Labeler")
        self.geometry("1300x900")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.current_candidates = []
        self.candidate_page = 0
        self.n_per_page = 6
        self._view_mode = "random"  # "top" | "low_conf" | "random"

        self._build_ui()
        self._update_status()

    # ── UI construction ──────────────────────────────────────────────────
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

        # ── Control panel (left sidebar) ──
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

        # Train
        ttk.Button(ctrl, text="Train Model", command=self._train).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Review predictions
        ttk.Label(ctrl, text="Review label:").pack(anchor=tk.W)
        self.review_label_var = tk.StringVar()
        self.review_combo = ttk.Combobox(
            ctrl, textvariable=self.review_label_var, state="readonly", width=20
        )
        self.review_combo.pack(fill=tk.X)
        self.unlabeled_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl, text="Unlabeled only", variable=self.unlabeled_only_var
        ).pack(anchor=tk.W)
        pred_row = ttk.Frame(ctrl)
        pred_row.pack(fill=tk.X, pady=2)
        ttk.Button(pred_row, text="Show Top", command=self._show_predictions).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(pred_row, text="Show Low", command=self._show_low_confidence).pack(
            side=tk.LEFT, expand=True, fill=tk.X
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

        # PELT Tuner
        ttk.Button(ctrl, text="PELT Tuner", command=self._show_pelt_tuner).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Diagnostics
        ttk.Button(ctrl, text="Diagnostics", command=self._show_diagnostics).pack(
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

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ── Channel legend ──
        legend_frame = ttk.LabelFrame(ctrl, text="Channel Legend", padding=4)
        legend_frame.pack(fill=tk.X, pady=2)
        for ch in self.labeler.channels:
            hex_color = mcolors.to_hex(self.labeler.channel_colors[ch])
            row = ttk.Frame(legend_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Canvas(
                row, width=14, height=14, bg=hex_color, highlightthickness=0
            ).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Label(row, text=ch).pack(side=tk.LEFT)

        # ── Log panel ──
        ttk.Label(ctrl, text="Log:").pack(anchor=tk.W, pady=(8, 0))
        self.log_text = tk.Text(
            ctrl,
            width=28,
            height=12,
            font=("Courier", 9),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ── Main area (trace gallery) ──
        self.gallery_frame = ttk.Frame(self)
        self.gallery_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

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
        self.status_var.set(
            f"Round {self.labeler.round_num}  |  "
            f"Labeled: {self.labeler.n_labeled}/{self.labeler.N}  |  "
            f"[{counts_str}]{hist}"
        )
        # update review combo
        opts = sorted(self.labeler.label_set) if self.labeler.label_set else []
        self.review_combo["values"] = opts
        if opts and self.review_label_var.get() not in opts:
            self.review_label_var.set(opts[0])

    # ── Actions ──────────────────────────────────────────────────────────
    def _add_label(self):
        name = self.new_label_var.get().strip()
        if not name:
            return
        self.labeler.label_set.add(name)
        self.new_label_var.set("")
        self._update_status()
        self._log(f"Added label: '{name}'")

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
        target = self.review_label_var.get()
        if not target or self.labeler.proba is None:
            messagebox.showinfo("Review", "Train the model first.")
            return
        unlabeled_only = self.unlabeled_only_var.get()
        self.current_candidates = self.labeler.get_top_predictions(
            target, n=50, unlabeled_only=unlabeled_only
        )
        if not self.current_candidates:
            self._log(f"No candidates for '{target}'")
            return
        self._view_mode = "top"
        self.candidate_page = 0
        self._render_page()

    def _show_low_confidence(self):
        target = self.review_label_var.get()
        if not target or self.labeler.proba is None:
            messagebox.showinfo("Review", "Train the model first.")
            return
        unlabeled_only = self.unlabeled_only_var.get()
        self.current_candidates = self.labeler.get_low_confidence_predictions(
            target, n=50, unlabeled_only=unlabeled_only
        )
        if not self.current_candidates:
            self._log(f"No low-confidence candidates for '{target}'")
            return
        self._view_mode = "low_conf"
        self.candidate_page = 0
        self._render_page()

    def _show_random(self):
        unlabeled_only = self.unlabeled_only_var.get()
        pairs = [(row.exp, int(row.idx)) for _, row in self.labeler.index_df.iterrows()]
        if unlabeled_only:
            pairs = [p for p in pairs if self.labeler.get_label(*p) == UNLABELED]
        if not pairs:
            self._log("No eligible traces to show.")
            return
        sample = random.sample(pairs, min(50, len(pairs)))
        self.current_candidates = [
            {
                "exp": e,
                "idx": i,
                "prob": None,
                "current_label": self.labeler.get_label(e, i),
            }
            for e, i in sample
        ]
        self._view_mode = "random"
        self.candidate_page = 0
        self._render_page()

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

    def _prev_page(self):
        if self.candidate_page > 0:
            self.candidate_page -= 1
            self._render_page()

    def _next_page(self):
        max_page = max(0, (len(self.current_candidates) - 1) // self.n_per_page)
        if self.candidate_page < max_page:
            self.candidate_page += 1
            self._render_page()

    def _confirm_all(self):
        target = self.review_label_var.get()
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

        is_random = self._view_mode == "random"
        target = None if is_random else self.review_label_var.get()
        start = self.candidate_page * self.n_per_page
        page = self.current_candidates[start : start + self.n_per_page]
        n_total = len(self.current_candidates)
        max_page = max(0, (n_total - 1) // self.n_per_page)
        page_str = f"Page {self.candidate_page + 1}/{max_page + 1}  ({n_total} total)"

        if self._view_mode == "random":
            header_text = f"Random traces  —  {page_str}"
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
        card = ttk.LabelFrame(parent, text=title, padding=4)

        # Matplotlib figure embedded in tk
        fig = Figure(figsize=(5.2, 2.0), dpi=90)
        ax = fig.add_subplot(111)
        title_extra = "" if is_random else f"  P({target_label})={prob:.2f}"
        self.labeler.plot_trace(exp, idx, ax=ax, title_extra=title_extra)
        fig.tight_layout(pad=0.4)
        self._card_figures.append(fig)
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button row
        btn_row = ttk.Frame(card)
        btn_row.pack(fill=tk.X, pady=(2, 0))

        all_labels = sorted(self.labeler.label_set)

        if is_random:
            # Random mode: dropdown over all labels + "Label" button
            label_var = tk.StringVar(value=all_labels[0] if all_labels else "")
            combo = ttk.Combobox(
                btn_row,
                textvariable=label_var,
                values=all_labels,
                state="readonly",
                width=16,
            )
            combo.pack(side=tk.LEFT, padx=2)
            ttk.Button(
                btn_row,
                text="Label",
                command=lambda e=exp, i=idx, rv=label_var: self._confirm_one(
                    e, i, rv.get(), card
                ),
            ).pack(side=tk.LEFT, padx=2)
        else:
            # Prediction mode: quick-confirm button for target label
            ttk.Button(
                btn_row,
                text=f"✓ {target_label}",
                command=lambda e=exp, i=idx, tl=target_label: self._confirm_one(
                    e, i, tl, card
                ),
            ).pack(side=tk.LEFT, padx=2)

            # Reassign dropdown for other labels
            other_labels = [l for l in all_labels if l != target_label]
            if other_labels:
                reassign_var = tk.StringVar(value=other_labels[0])
                combo = ttk.Combobox(
                    btn_row,
                    textvariable=reassign_var,
                    values=other_labels,
                    state="readonly",
                    width=14,
                )
                combo.pack(side=tk.LEFT, padx=2)
                ttk.Button(
                    btn_row,
                    text="Assign",
                    command=lambda e=exp, i=idx, rv=reassign_var: self._reassign_one(
                        e, i, rv.get(), card
                    ),
                ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_row,
            text="✕ Unlabel",
            command=lambda e=exp, i=idx: self._unlabel_one(e, i, card),
        ).pack(side=tk.RIGHT, padx=2)
        skip_btn = ttk.Button(
            btn_row, text="Skip", command=lambda: self._skip_card(card)
        )
        skip_btn.pack(side=tk.RIGHT, padx=2)

        cur = cand["current_label"]
        cur_str = cur if cur != UNLABELED else "unlabeled"
        ttk.Label(btn_row, text=f"  cur: {cur_str}", font=("Helvetica", 8)).pack(
            side=tk.RIGHT
        )

        self.card_widgets.append(card)
        return card

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
        target = self.review_label_var.get()
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

    def _skip_card(self, card):
        self._flash_card(card, "gray")

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
        self._update_status()
        self._log(f"Loaded {n} labels from {path}")

    def _on_close(self):
        if self.labeler.n_labeled > 0:
            if messagebox.askyesno("Quit", "Save labels before closing?"):
                self._save()
        self.destroy()


def LabelerApp(labeler):
    """Factory: creates a tk.Tk subclass with all LabelerApp methods."""

    class _App(_LabelerAppMethods, tk.Tk):
        pass

    return _App(labeler)
