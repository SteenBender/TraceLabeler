"""Feature extraction from tracks and PELT results."""

from itertools import combinations

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
    # Generic pairwise cross-channel features for every pair of channels.
    for ch_i, ch_j in combinations(channels, 2):
        n_i = feats.get(f"{ch_i}_n_pos_steps", 0)
        n_j = feats.get(f"{ch_j}_n_pos_steps", 0)
        # Step count and molecule ratios
        feats[f"{ch_i}_{ch_j}_step_ratio"] = (
            (n_i / n_j) if n_j > 0 else (float("inf") if n_i > 0 else 0.0)
        )
        feats[f"{ch_i}_{ch_j}_mol_ratio"] = feats.get(f"{ch_i}_mol_pos", 0) / (
            feats.get(f"{ch_j}_mol_pos", 0) + 1e-6
        )
        # First positive step lag: positive = ch_i leads ch_j, negative = ch_j leads
        fp_i = feats.get(f"{ch_i}_first_pos_frame", np.nan)
        fp_j = feats.get(f"{ch_j}_first_pos_frame", np.nan)
        feats[f"{ch_i}_{ch_j}_first_step_lag"] = (
            float(fp_i - fp_j) if not (np.isnan(fp_i) or np.isnan(fp_j)) else np.nan
        )
        # Pearson correlation of raw signals
        sig_i = tr[ch_i].astype(float)
        sig_j = tr[ch_j].astype(float)
        if sig_i.std() > 0 and sig_j.std() > 0:
            feats[f"{ch_i}_{ch_j}_signal_corr"] = float(np.corrcoef(sig_i, sig_j)[0, 1])
        else:
            feats[f"{ch_i}_{ch_j}_signal_corr"] = np.nan
        # Pearson correlation of PELT fits (captures co-varying step structure)
        fit_i = nr[ch_i]["fit"].astype(float)
        fit_j = nr[ch_j]["fit"].astype(float)
        if fit_i.std() > 0 and fit_j.std() > 0:
            feats[f"{ch_i}_{ch_j}_fit_corr"] = float(np.corrcoef(fit_i, fit_j)[0, 1])
        else:
            feats[f"{ch_i}_{ch_j}_fit_corr"] = np.nan
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
