"""Global configuration — paths, channel names, calibration constants."""

import os
from pathlib import Path

# Data paths — override via environment variables or --data-root CLI flag.
DATA_ROOT = Path(os.environ.get("CLATHRIN_DATA_ROOT", "Uncoating"))
# Root for raw TIFF files (local path; C1/C2/C3 per channel per experiment)
TIFF_ROOT = Path(os.environ.get("CLATHRIN_TIFF_ROOT", str(DATA_ROOT)))
EXPERIMENTS = [f"Ex{i:02d}" for i in range(1, 8)]
CHANNELS = ["clathrin", "hsc70", "auxilin"]
CHANNEL_COLORS = {"clathrin": "magenta", "hsc70": "cyan", "auxilin": "yellow"}
# Single-molecule intensity calibration (AU per molecule). Channels not listed
# are left unscaled (factor = 1.0). Plotted y-axis becomes "molecules".
CALIBRATION = {"clathrin": 1.0, "hsc70": 13.3, "auxilin": 10.0}
SAVE_PATH = "labels/labels.json"
PICKLE_PATH = "results/trace_data.pkl"  # pre-computed data cache

N_JOBS = min(os.cpu_count() or 4, 8)  # joblib parallelism for PELT fitting

# Default palette used when assigning colors to newly detected channels.
_DEFAULT_COLORS = [
    "magenta",
    "cyan",
    "yellow",
    "lime",
    "orange",
    "red",
    "white",
    "blue",
]


def configure_channels(names, colors=None, calibration=None):
    """Update runtime channel configuration in-place.

    Parameters
    ----------
    names : list[str]
        Ordered channel names (index 0 = column 0 in the A matrix).
    colors : dict[str, str] | None
        Mapping name → matplotlib color string. Missing entries get defaults.
    calibration : dict[str, float] | None
        Mapping name → AU-per-molecule scale factor. Missing entries default to 1.0.
    """
    global CHANNELS, CHANNEL_COLORS, CALIBRATION
    CHANNELS = list(names)
    CHANNEL_COLORS = {}
    for i, ch in enumerate(CHANNELS):
        if colors and ch in colors:
            CHANNEL_COLORS[ch] = colors[ch]
        else:
            CHANNEL_COLORS[ch] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
    CALIBRATION = {ch: float((calibration or {}).get(ch, 1.0)) for ch in CHANNELS}


def detect_n_channels(data_root, experiments=None):
    """Scan ProcessedTracks.mat in the first available experiment and return
    the number of signal channels (columns of the A matrix).

    Returns None if no .mat file could be read.
    """
    import scipy.io as sio

    exps = experiments or EXPERIMENTS
    for exp in exps:
        pt_path = Path(data_root) / exp / "ProcessedTracks.mat"
        if not pt_path.exists():
            continue
        try:
            pt = sio.loadmat(str(pt_path))
            tracks_raw = pt["tracks"]
            if tracks_raw.shape[1] == 0:
                continue
            A = tracks_raw[0, 0]["A"]
            return A.shape[0]  # rows = channels
        except Exception:
            continue
    return None
