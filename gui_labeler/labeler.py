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
        feature_mode="pelt",
        cnn_device="cpu",
    ):
        self.all_data = all_data
        self.new_results = new_results
        self.channels = channels
        self.channel_colors = channel_colors
        self.calibration = calibration or {}
        self.experiments = experiments or sorted(all_data.keys())
        self.save_path = save_path or config.SAVE_PATH
        self.feature_mode = feature_mode  # "pelt" or "pretrained"
        self.cnn_device = cnn_device
        self.pkl_path = (
            None  # set by __main__ after loading; used to sync pickle on refit
        )

        print("Extracting PELT features from all tracks …")
        self.feat_df, self.index_df = build_feature_matrix(
            all_data, new_results, self.experiments, channels
        )
        self.N = len(self.feat_df)
        print(f"  {self.N} tracks, {self.feat_df.shape[1]} features each")

        # O(1) lookup: (exp, idx) → global row index
        self._global_idx_map = {
            (row.exp, int(row.idx)): gi for gi, row in self.index_df.iterrows()
        }

        # Build raw trace sequences for pretrained features: list of (C, T_i) arrays
        self._sequences = []
        for exp in self.experiments:
            for idx in range(len(all_data[exp])):
                tr = all_data[exp][idx]
                seq = np.stack([tr[ch].astype(np.float64) for ch in channels], axis=0)
                self._sequences.append(seq)

        self.labels = {}
        self.label_set = set()
        self.model = None
        self._encoder = None  # TraceEncoder instance (persists across rounds)
        self._embed_features = None  # cached (N, embed_dim) numpy array
        self.proba = None
        self.classes = None
        self.round_num = 0
        self.history = []

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

    # ── CNN pretraining & embedding extraction ──
    def pretrain_cnn(self):
        """Run MAE pretraining on all traces and cache embeddings."""
        from .model import TraceEncoder

        if self._encoder is not None and self._encoder.is_pretrained:
            if self._embed_features is None:
                # Encoder trained but embeddings not yet extracted (e.g. race
                # condition between background pretrain thread and train()).
                print("[Encoder] Extracting embeddings (encoder was already pretrained) …")
                self._embed_features = self._encoder.extract_embeddings(self._sequences)
                print(f"  Embedding shape: {self._embed_features.shape}")
            return "Encoder already pretrained."
        self._encoder = TraceEncoder(
            n_channels=len(self.channels), device=self.cnn_device
        )
        print("[Encoder] Pretraining masked autoencoder on all traces …")
        self._encoder.pretrain(self._sequences)
        print("[Encoder] Extracting embeddings …")
        self._embed_features = self._encoder.extract_embeddings(self._sequences)
        print(f"  Embedding shape: {self._embed_features.shape}")
        return "Encoder pretrained, embeddings cached."

    # ── train ──
    def train(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        labeled_mask = self.get_labeled_mask()
        label_arr = self.get_label_array()
        y_labeled = label_arr[labeled_mask]
        if len(set(y_labeled)) < 2:
            return None, "Need ≥ 2 distinct labels."

        # Select feature matrix based on mode
        if self.feature_mode == "pretrained":
            if self._embed_features is None:
                self.pretrain_cnn()
            X_all = self._embed_features
        else:
            X_all = self.feat_df.values

        X_labeled = X_all[labeled_mask]

        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.model.fit(X_labeled, y_labeled)
        self.classes = self.model.classes_

        cv_mean = float("nan")
        if len(X_labeled) >= 10:
            n_splits = min(5, len(X_labeled) // max(len(set(y_labeled)), 1))
            n_splits = max(2, n_splits)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    self.model, X_labeled, y_labeled, cv=n_splits, scoring="accuracy"
                )
            cv_mean = cv_scores.mean()

        self.proba = self.model.predict_proba(X_all)
        self.round_num += 1
        train_acc = self.model.score(X_labeled, y_labeled)
        mode_tag = "pretrained" if self.feature_mode == "pretrained" else "PELT"
        self.history.append(
            {
                "round": self.round_num,
                "n_labeled": self.n_labeled,
                "train_acc": train_acc,
                "cv_acc": cv_mean,
                "label_counts": self.label_counts(),
                "feature_mode": self.feature_mode,
            }
        )
        msg = (
            f"Round {self.round_num} [{mode_tag}] features={X_all.shape[1]}d: "
            f"{self.n_labeled} labeled ({self.label_counts()})\n"
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
        primary_ch = self.channels[0]
        for ch in self.channels:
            color = self.channel_colors[ch]
            scale = self.calibration.get(ch, 1.0)
            target_ax = ax if ch == primary_ch else ax2
            target_ax.plot(t, tr[ch] / scale, color=color, alpha=0.45, lw=0.5)
            target_ax.plot(t, nr[ch]["fit"] / scale, color=color, lw=1.3)
        ax.set_ylabel(primary_ch, fontsize=7, color=self.channel_colors[primary_ch])
        ax.tick_params(
            axis="y", labelsize=7, labelcolor=self.channel_colors[primary_ch]
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
