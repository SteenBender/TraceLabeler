"""Raw .mat data loading and parallel PELT pipeline."""

import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed

from . import config
from .pelt import fit_track_pelt


def load_experiment(exp_name):
    """Load ProcessedTracks and step fits for one experiment."""
    exp_dir = config.DATA_ROOT / exp_name
    if not exp_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}\n"
            f"Check that --data-root points to the folder containing ExNN/ directories."
        )
    pt_path = exp_dir / "ProcessedTracks.mat"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing {pt_path}")
    pt = sio.loadmat(str(pt_path))
    tracks_raw = pt["tracks"]

    steps_raw = {}
    for ch in config.CHANNELS:
        s = sio.loadmat(str(exp_dir / f"steps_{ch}.mat"))
        steps_raw[ch] = s[f"steps_{ch}"]

    n_tracks = tracks_raw.shape[1]
    tracks = []
    for i in range(n_tracks):
        tr = tracks_raw[0, i]
        t = tr["t"].flatten()
        f = tr["f"].flatten()
        A = tr["A"]

        track_dict = {
            "index": i,
            "time": t,
            "frames": f,
            "clathrin": A[0],
            "hsc70": A[1],
            "auxilin": A[2],
            "lifetime_s": float(tr["lifetime_s"].flat[0]),
            "x": tr["x"][0],
            "y": tr["y"][0],
        }

        for ch_idx, ch in enumerate(config.CHANNELS):
            sr = steps_raw[ch][0, i]
            ns = int(sr["numSteps"].flat[0]) if sr["numSteps"].size > 0 else 0
            track_dict[f"{ch}_fit"] = (
                sr["fit"].flatten() if sr["fit"].size > 0 else np.array([])
            )
            track_dict[f"{ch}_numSteps"] = ns
            track_dict[f"{ch}_stepStart"] = (
                sr["stepStart"].flatten() if ns > 0 else np.array([])
            )
            track_dict[f"{ch}_stepSize"] = (
                sr["stepSize"].flatten() if ns > 0 else np.array([])
            )

        tracks.append(track_dict)
    return tracks


def _load_and_pelt():
    """Load raw .mat data and run PELT.  Returns (all_data, pelt_results)."""
    print("[1/3] Loading experiments …")
    all_data = {}
    for exp in config.EXPERIMENTS:
        all_data[exp] = load_experiment(exp)
        print(f"  {exp}: {len(all_data[exp])} tracks")

    print("[2/3] Running PELT step detection …")
    pelt_results = {}
    for exp in config.EXPERIMENTS:
        tracks = all_data[exp]
        print(f"  {exp}: {len(tracks)} tracks …", end=" ", flush=True)
        results = Parallel(n_jobs=config.N_JOBS)(
            delayed(fit_track_pelt)(tr) for tr in tracks
        )
        pelt_results[exp] = results
        n_steps = [r["clathrin"]["n_steps"] for r in results]
        print(f"done (median clathrin steps: {np.median(n_steps):.0f})")
    return all_data, pelt_results
