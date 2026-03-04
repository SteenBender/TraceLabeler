"""Feature extraction from tracks and PELT results."""

import numpy as np
import pandas as pd


def get_pelt_steps(nr, ch):
    fit = nr[ch]["fit"]
    step_frames = np.asarray(nr[ch]["step_frames"], dtype=int)
    if len(step_frames) == 0:
        return np.array([], dtype=int), np.array([])
    sizes = np.array([float(fit[sf] - fit[sf - 1]) for sf in step_frames])
    return step_frames, sizes


def extract_features(tr, nr, channels):
    t = tr["time"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.2
    T = len(t)
    feats = {}
    for ch in channels:
        signal = tr[ch]
        fit = nr[ch]["fit"]
        sf, sz = get_pelt_steps(nr, ch)
        sig_range = signal.max() - signal.min() + 1e-6
        thresh = 0.05 * sig_range
        pos = sz[sz > thresh]
        neg = sz[sz < -thresh]
        feats[f"{ch}_n_pos_steps"] = len(pos)
        feats[f"{ch}_n_neg_steps"] = len(neg)
        feats[f"{ch}_n_total_steps"] = nr[ch]["n_steps"]
        feats[f"{ch}_mol_pos"] = float(pos.sum()) if len(pos) else 0.0
        feats[f"{ch}_mol_neg"] = float(-neg.sum()) if len(neg) else 0.0
        feats[f"{ch}_mean"] = float(signal.mean())
        feats[f"{ch}_std"] = float(signal.std())
        feats[f"{ch}_max"] = float(signal.max())
        feats[f"{ch}_min"] = float(signal.min())
        feats[f"{ch}_range"] = float(sig_range)
        feats[f"{ch}_initial"] = float(fit[0])
        feats[f"{ch}_final"] = float(fit[-1])
        feats[f"{ch}_net_change"] = float(fit[-1] - fit[0])
        feats[f"{ch}_med_pos_step"] = float(np.median(pos)) if len(pos) > 0 else 0.0
        pos_frames = sf[sz > thresh]
        feats[f"{ch}_first_pos_frame"] = (
            float(pos_frames[0] * dt) if len(pos_frames) > 0 else np.nan
        )
    feats["lifetime"] = float(tr["lifetime_s"])
    feats["n_frames"] = T
    h_pos = feats.get("hsc70_n_pos_steps", 0)
    a_pos = feats.get("auxilin_n_pos_steps", 0)
    feats["hsc70_aux_step_ratio"] = (
        (h_pos / a_pos) if a_pos > 0 else (float("inf") if h_pos > 0 else 0.0)
    )
    feats["hsc70_aux_mol_ratio"] = feats.get("hsc70_mol_pos", 0) / (
        feats.get("auxilin_mol_pos", 0) + 1e-6
    )
    return feats


def build_feature_matrix(all_data, new_results, experiments, channels):
    feat_rows, index_rows = [], []
    for exp in experiments:
        for idx in range(len(all_data[exp])):
            tr = all_data[exp][idx]
            nr = new_results[exp][idx]
            feats = extract_features(tr, nr, channels)
            feat_rows.append(feats)
            index_rows.append({"exp": exp, "idx": idx})
    feat_df = pd.DataFrame(feat_rows)
    index_df = pd.DataFrame(index_rows)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    return feat_df, index_df
