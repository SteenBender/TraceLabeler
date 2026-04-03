"""Pickle-based save/load for pre-computed data (server-side preparation)."""

import os
import pickle

from . import config
from .data_loader import _load_and_pelt


def save_prepared_data(out_path=None):
    """Load .mat data, run PELT, save everything to a single pickle file.
    Run this on the server (no display needed):

        python -m gui_labeler --prepare
    """
    out_path = out_path or config.PICKLE_PATH
    all_data, pelt_results = _load_and_pelt()

    bundle = {
        "all_data": all_data,
        "pelt_results": pelt_results,
        "experiments": config.EXPERIMENTS,
        "channels": config.CHANNELS,
        "channel_colors": config.CHANNEL_COLORS,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved prepared data \u2192 {out_path}  ({size_mb:.1f} MB)")
    print(
        "Now run the GUI:\n"
        f"  python -m gui_labeler              "
        f"  (looks for {out_path} in cwd)"
    )


def save_bundle(all_data, pelt_results, out_path):
    """Write already-computed data back to a pickle (e.g. after re-fitting PELT)."""
    bundle = {
        "all_data": all_data,
        "pelt_results": pelt_results,
        "experiments": config.EXPERIMENTS,
        "channels": config.CHANNELS,
        "channel_colors": config.CHANNEL_COLORS,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / 1e6
    return size_mb


def load_prepared_data(pkl_path=None):
    """Load pre-computed data from pickle.  Returns (all_data, pelt_results, stored_channels)."""
    pkl_path = pkl_path or config.PICKLE_PATH
    print(f"Loading pre-computed data from {pkl_path} \u2026")
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    all_data = bundle["all_data"]
    pelt_results = bundle["pelt_results"]
    stored_channels = bundle.get("channels", config.CHANNELS)
    n_tracks = sum(len(v) for v in all_data.values())
    print(f"  {len(all_data)} experiments, {n_tracks} tracks total")
    return all_data, pelt_results, stored_channels
