"""Active-learning trace labeler — GUI-agnostic core."""

import json
import os
import warnings

import numpy as np
import pandas as pd

from . import config
from .features import build_feature_matrix

UNLABELED = "__unlabeled__"


def _json_default(obj):
    """Allow json.dump to serialize numpy scalars and arrays."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ActiveLabeler:
    """Active-learning trace labeler — GUI-agnostic core."""

    def __init__(
        self,
        all_data,
        new_results,
        channels,
        channel_colors,
        experiments=None,
        save_path=None,
        calibration=None,
    ):
        self.all_data = all_data
        self.new_results = new_results
        self.channels = channels
        self.channel_colors = channel_colors
        self.calibration = calibration or {}
        self.experiments = experiments or sorted(all_data.keys())
        self.save_path = save_path or config.SAVE_PATH
        self.pkl_path = (
            None  # set by __main__ after loading; used to sync pickle on refit
        )

        print("Extracting features from all tracks …")
        self.feat_df, self.index_df = build_feature_matrix(
            all_data, new_results, self.experiments, channels
        )
        self.N = len(self.feat_df)
        print(f"  {self.N} tracks, {self.feat_df.shape[1]} features each")

        # O(1) lookup: (exp, idx) → global row index
        self._global_idx_map = {
            (row.exp, int(row.idx)): gi for gi, row in self.index_df.iterrows()
        }

        self.labels = {}
        self.label_set = set()
        self.model = None
        from sklearn.preprocessing import RobustScaler

        self.scaler = RobustScaler()
        self.proba = None
        self.classes = None
        self.round_num = 0
        self.history = []
        self.df_triggered = None

    # ── helpers ──
    def _key(self, exp, idx):
        return f"{exp}:{idx}"

    def _parse_key(self, key):
        exp, idx = key.rsplit(":", 1)
        return exp, int(idx)

    def _global_idx(self, exp, idx):
        return self._global_idx_map.get((exp, idx))

    # ── label mgmt ──
    def set_label(self, exp, idx, label):
        self.labels[self._key(exp, idx)] = label
        self.label_set.add(label)

    def get_label(self, exp, idx):
        return self.labels.get(self._key(exp, idx), UNLABELED)

    def remove_label(self, exp, idx):
        self.labels.pop(self._key(exp, idx), None)

    @property
    def n_labeled(self):
        return len(self.labels)

    def label_counts(self):
        if not self.labels:
            return {}
        return dict(pd.Series(list(self.labels.values())).value_counts())

    def get_labeled_mask(self):
        mask = np.zeros(self.N, dtype=bool)
        for key in self.labels:
            exp, idx = self._parse_key(key)
            gi = self._global_idx(exp, idx)
            if gi is not None:
                mask[gi] = True
        return mask

    def get_label_array(self):
        arr = np.full(self.N, UNLABELED, dtype=object)
        for key, label in self.labels.items():
            exp, idx = self._parse_key(key)
            gi = self._global_idx(exp, idx)
            if gi is not None:
                arr[gi] = label
        return arr

    def seed_from_analytical(
        self,
        df_triggered,
        excess_col="is_excess_mol",
        label_excess="hsc70_excess",
        label_other="balanced",
    ):
        n_seeded = 0
        for _, row in df_triggered.iterrows():
            exp, idx = row.exp, int(row.idx)
            label = label_excess if row[excess_col] else label_other
            self.set_label(exp, idx, label)
            n_seeded += 1
        return n_seeded

    # ── train ──
    def train(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        labeled_mask = self.get_labeled_mask()
        label_arr = self.get_label_array()
        X_all = self.feat_df.values
        X_labeled = X_all[labeled_mask]
        y_labeled = label_arr[labeled_mask]
        if len(set(y_labeled)) < 2:
            return None, "Need ≥ 2 distinct labels."
        # HistGradientBoostingClassifier handles NaN natively — no
        # scaler/imputation needed, but we keep the scaler for any
        # downstream code that might reference it.
        self.scaler.fit(X_labeled)
        X_scaled = X_labeled  # raw features (NaN-safe)
        X_all_scaled = X_all
        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.model.fit(X_scaled, y_labeled)
        self.classes = self.model.classes_
        if len(X_scaled) >= 10:
            n_splits = min(5, len(X_scaled) // max(len(set(y_labeled)), 1))
            n_splits = max(2, n_splits)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    self.model, X_scaled, y_labeled, cv=n_splits, scoring="accuracy"
                )
            cv_mean = cv_scores.mean()
        else:
            cv_mean = float("nan")
        self.proba = self.model.predict_proba(X_all_scaled)
        self.round_num += 1
        train_acc = self.model.score(X_scaled, y_labeled)
        self.history.append(
            {
                "round": self.round_num,
                "n_labeled": self.n_labeled,
                "train_acc": train_acc,
                "cv_acc": cv_mean,
                "label_counts": self.label_counts(),
            }
        )
        msg = (
            f"Round {self.round_num}: {self.n_labeled} labeled "
            f"({self.label_counts()})\n"
            f"  Train acc: {train_acc:.3f}  |  CV acc: {cv_mean:.3f}"
        )
        return self.model, msg

    def get_top_predictions(self, target_label, n=20, unlabeled_only=True):
        if self.proba is None:
            return []
        try:
            label_idx = list(self.classes).index(target_label)
        except ValueError:
            return []  # target_label was not present in the training set
        probs = self.proba[:, label_idx]
        labeled_mask = self.get_labeled_mask()
        label_arr = self.get_label_array()
        eligible = ~labeled_mask if unlabeled_only else np.ones(self.N, dtype=bool)
        order = np.argsort(-probs)
        results = []
        for gi in order:
            if not eligible[gi]:
                continue
            exp = self.index_df.iloc[gi].exp
            idx = int(self.index_df.iloc[gi].idx)
            results.append(
                {
                    "exp": exp,
                    "idx": idx,
                    "global_idx": int(gi),
                    "prob": float(probs[gi]),
                    "current_label": label_arr[gi],
                }
            )
            if len(results) >= n:
                break
        return results

    def get_low_confidence_predictions(self, target_label, n=50, unlabeled_only=True):
        """Return traces predicted as *target_label* with the lowest confidence.

        These are traces where the model's top predicted class equals
        *target_label*, sorted by ascending probability — i.e. the most
        uncertain predictions for that class.
        """
        if self.proba is None:
            return []
        try:
            label_idx = list(self.classes).index(target_label)
        except ValueError:
            return []  # target_label was not present in the training set
        probs = self.proba[:, label_idx]
        predicted_classes = self.classes[np.argmax(self.proba, axis=1)]
        labeled_mask = self.get_labeled_mask()
        label_arr = self.get_label_array()
        eligible = ~labeled_mask if unlabeled_only else np.ones(self.N, dtype=bool)
        # Only keep traces whose *predicted* class is the target
        predicted_as_target = predicted_classes == target_label
        mask = eligible & predicted_as_target
        # Sort by ascending probability (lowest confidence first)
        order = np.argsort(probs)
        results = []
        for gi in order:
            if not mask[gi]:
                continue
            exp = self.index_df.iloc[gi].exp
            idx = int(self.index_df.iloc[gi].idx)
            results.append(
                {
                    "exp": exp,
                    "idx": idx,
                    "global_idx": int(gi),
                    "prob": float(probs[gi]),
                    "current_label": label_arr[gi],
                }
            )
            if len(results) >= n:
                break
        return results

    # ── plotting helpers ──
    def plot_trace(self, exp, idx, ax=None, title_extra=""):
        import matplotlib.pyplot as plt

        tr = self.all_data[exp][idx]
        nr = self.new_results[exp][idx]
        t = tr["time"]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        ax2 = ax.twinx()
        for ch in self.channels:
            color = self.channel_colors[ch]
            scale = self.calibration.get(ch, 1.0)
            target_ax = ax if ch == "clathrin" else ax2
            target_ax.plot(t, tr[ch] / scale, color=color, alpha=0.45, lw=0.5)
            target_ax.plot(t, nr[ch]["fit"] / scale, color=color, lw=1.3)
        ax.set_ylabel("clathrin", fontsize=7, color=self.channel_colors["clathrin"])
        ax.tick_params(
            axis="y", labelsize=7, labelcolor=self.channel_colors["clathrin"]
        )
        ax2.set_ylabel("molecules", fontsize=7)
        ax2.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        label = self.get_label(exp, idx)
        label_str = label if label != UNLABELED else "unlabeled"
        ax.set_title(f"{exp} #{idx}  [{label_str}] {title_extra}", fontsize=9)
        return ax

    # ── save / load ──
    def save(self, path=None):
        import tempfile

        path = path or self.save_path
        data = {
            "labels": self.labels,
            "label_set": sorted(self.label_set),
            "round_num": self.round_num,
            "history": self.history,
        }
        dir_ = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=_json_default)
            os.replace(tmp, path)  # atomic on POSIX
        except Exception:
            os.unlink(tmp)
            raise
        return path

    def load(self, path=None):
        path = path or self.save_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Labels file not found: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted labels file '{path}': {exc}") from exc
        self.labels = data["labels"]
        self.label_set = set(data.get("label_set", []))
        self.round_num = data.get("round_num", 0)
        self.history = data.get("history", [])
        self.label_set.update(set(self.labels.values()))
        return len(self.labels)
