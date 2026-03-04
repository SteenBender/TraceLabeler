"""Analytical Auxilin-triggered Hsc70 excess detection."""

import numpy as np
import pandas as pd

from .features import get_pelt_steps


def analyze_auxilin_triggered_hsc70(tr, nr, min_step_frac=0.05, mol_quanta=None):
    h_frames, h_sizes = get_pelt_steps(nr, "hsc70")
    a_frames, a_sizes = get_pelt_steps(nr, "auxilin")
    h_range = tr["hsc70"].max() - tr["hsc70"].min() + 1e-6
    a_range = tr["auxilin"].max() - tr["auxilin"].min() + 1e-6
    h_thresh = min_step_frac * h_range
    a_thresh = min_step_frac * a_range
    h_pos_mask = h_sizes > h_thresh
    a_pos_mask = a_sizes > a_thresh
    h_pos_frames = h_frames[h_pos_mask]
    h_pos_sizes = h_sizes[h_pos_mask]
    a_pos_frames = a_frames[a_pos_mask]
    a_pos_sizes = a_sizes[a_pos_mask]

    base = dict(
        first_aux_frame=-1,
        first_hsc70_frame=-1,
        hsc70_after_steps=0,
        aux_after_steps=0,
        hsc70_after_mol=0.0,
        aux_after_mol=0.0,
        hsc70_after_mol_n=0.0,
        aux_after_mol_n=0.0,
        step_ratio=np.nan,
        mol_ratio=np.nan,
        is_excess_steps=False,
        is_excess_mol=False,
        h_quanta=np.nan,
        a_quanta=np.nan,
    )

    if len(a_pos_frames) == 0:
        return base
    first_aux_frame = int(a_pos_frames[0])

    h_after_aux = h_pos_frames[h_pos_frames >= first_aux_frame]
    if len(h_after_aux) == 0:
        base["first_aux_frame"] = first_aux_frame
        return base
    first_hsc70_frame = int(h_after_aux[0])

    if mol_quanta is not None:
        h_quanta = a_quanta = mol_quanta
    else:
        h_quanta = float(np.median(h_pos_sizes)) if len(h_pos_sizes) > 0 else 1.0
        a_quanta = float(np.median(a_pos_sizes)) if len(a_pos_sizes) > 0 else 1.0
    h_quanta = max(h_quanta, 0.1)
    a_quanta = max(a_quanta, 0.1)

    h_subseq_mask = h_pos_frames > first_hsc70_frame
    a_subseq_mask = a_pos_frames > first_hsc70_frame
    n_h = int(h_subseq_mask.sum())
    n_a = int(a_subseq_mask.sum())
    mol_h = float(h_pos_sizes[h_subseq_mask].sum())
    mol_a = float(a_pos_sizes[a_subseq_mask].sum())
    mol_n_h = (
        float(np.round(h_pos_sizes[h_subseq_mask] / h_quanta).sum()) if n_h > 0 else 0.0
    )
    mol_n_a = (
        float(np.round(a_pos_sizes[a_subseq_mask] / a_quanta).sum()) if n_a > 0 else 0.0
    )
    step_ratio = (n_h / n_a) if n_a > 0 else (np.inf if n_h > 0 else np.nan)
    mol_ratio = (mol_h / mol_a) if mol_a > 0 else (np.inf if mol_h > 0 else np.nan)
    is_excess_steps = n_h > n_a
    is_excess_mol = mol_n_h > mol_n_a

    return dict(
        first_aux_frame=first_aux_frame,
        first_hsc70_frame=first_hsc70_frame,
        hsc70_after_steps=n_h,
        aux_after_steps=n_a,
        hsc70_after_mol=mol_h,
        aux_after_mol=mol_a,
        hsc70_after_mol_n=mol_n_h,
        aux_after_mol_n=mol_n_a,
        step_ratio=step_ratio,
        mol_ratio=mol_ratio,
        is_excess_steps=is_excess_steps,
        is_excess_mol=is_excess_mol,
        h_quanta=h_quanta,
        a_quanta=a_quanta,
    )


def run_analytical(all_data, new_results, experiments):
    records = []
    for exp in experiments:
        for idx, (tr, nr) in enumerate(zip(all_data[exp], new_results[exp])):
            res = analyze_auxilin_triggered_hsc70(tr, nr)
            res.update(exp=exp, idx=idx)
            records.append(res)
    df = pd.DataFrame(records)
    has_trigger = (df.first_aux_frame >= 0) & (df.first_hsc70_frame >= 0)
    df_triggered = df[has_trigger].reset_index(drop=True)
    return df_triggered
