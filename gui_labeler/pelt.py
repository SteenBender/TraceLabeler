"""PELT changepoint detection for intensity traces."""

import numpy as np

from . import config


def _merge_short_segments(signal, segments, min_len):
    segments = list(segments)
    while True:
        lengths = [e - s for s, e in segments]
        if len(segments) <= 1 or min(lengths) >= min_len:
            break
        idx = int(np.argmin(lengths))
        s_short, e_short = segments[idx]
        mean_short = np.mean(signal[s_short:e_short])
        candidates = []
        if idx > 0:
            s_l, e_l = segments[idx - 1]
            candidates.append((idx - 1, abs(np.mean(signal[s_l:e_l]) - mean_short)))
        if idx < len(segments) - 1:
            s_r, e_r = segments[idx + 1]
            candidates.append((idx + 1, abs(np.mean(signal[s_r:e_r]) - mean_short)))
        merge_into = min(candidates, key=lambda x: x[1])[0]
        s_m, e_m = segments[merge_into]
        new_s = min(s_short, s_m)
        new_e = max(e_short, e_m)
        lo, hi = min(idx, merge_into), max(idx, merge_into)
        segments[lo : hi + 1] = [(new_s, new_e)]
    return segments


def fit_pelt(signal, pen_mult=1.0, min_plateau=5):
    """Fit PELT changepoint model to a 1-D intensity signal.

    Parameters
    ----------
    pen_mult : float
        Scales the auto-computed penalty. Values > 1 → fewer breakpoints; < 1 → more.
    """
    import ruptures as rpt

    signal = np.asarray(signal, dtype=np.float64)
    T = len(signal)
    mad = np.median(np.abs(np.diff(signal))) / 0.6745
    pen = pen_mult * 3.0 * mad**2 * np.log(T)
    algo = rpt.Pelt(model="l2", min_size=max(2, min_plateau), jump=5)
    algo.fit(signal)
    # Run with the computed penalty. If ruptures raises ValueError (signal is so
    # noisy the penalty is effectively zero), nudge it upward until it succeeds.
    try:
        breakpoints = algo.predict(pen=pen)
    except ValueError:
        breakpoints = None
        for fallback in [2.0, 5.0]:
            try:
                breakpoints = algo.predict(pen=pen * fallback)
                break
            except ValueError:
                continue
    if breakpoints is None:
        return {
            "fit": np.full(T, np.mean(signal)),
            "states": np.zeros(T, dtype=int),
            "n_states": 1,
            "n_steps": 0,
            "step_frames": np.array([], dtype=int),
        }
    starts = [0] + breakpoints[:-1]
    ends = breakpoints
    segments = list(zip(starts, ends))
    if min_plateau > 1 and len(segments) > 1:
        segments = _merge_short_segments(signal, segments, min_plateau)
    fit = np.empty(T)
    state_labels = np.empty(T, dtype=int)
    for sid, (s, e) in enumerate(segments):
        fit[s:e] = np.mean(signal[s:e])
        state_labels[s:e] = sid
    sig_std = np.std(signal)
    tol = sig_std * 0.3 if sig_std > 0 else 1.0
    unique_means = []
    state_map = {}
    for sid in range(len(segments)):
        s, e = segments[sid]
        m = np.mean(signal[s:e])
        matched = False
        for uid, um in enumerate(unique_means):
            if abs(m - um) < tol:
                state_map[sid] = uid
                matched = True
                break
        if not matched:
            state_map[sid] = len(unique_means)
            unique_means.append(m)
    states = np.array([state_map[state_labels[i]] for i in range(T)])
    transitions = np.where(np.diff(states) != 0)[0]
    return {
        "fit": fit,
        "states": states,
        "n_states": len(set(states)),
        "n_steps": len(transitions),
        "step_frames": transitions + 1,
    }


def fit_track_pelt(tr, pen_mult=1.0, min_plateau=5, ch_pen_mult=None, ch_min_plateau=None, channels=None):
    """Fit PELT on each channel.

    ch_pen_mult / ch_min_plateau are optional dicts {channel_name: value}
    for per-channel overrides; missing channels fall back to the global scalars.
    channels must be passed explicitly when called from joblib workers, because
    subprocess workers import config fresh and won't see runtime mutations.
    """
    result = {}
    for ch in (channels or config.CHANNELS):
        pm = ch_pen_mult.get(ch, pen_mult) if ch_pen_mult else pen_mult
        mp = ch_min_plateau.get(ch, min_plateau) if ch_min_plateau else min_plateau
        result[ch] = fit_pelt(tr[ch], pen_mult=pm, min_plateau=mp)
    return result
