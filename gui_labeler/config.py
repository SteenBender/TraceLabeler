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
CALIBRATION = {"hsc70": 13.3, "auxilin": 10.0}
SAVE_PATH = "labels/labels.json"
PICKLE_PATH = "results/trace_data.pkl"  # pre-computed data cache

N_JOBS = min(os.cpu_count() or 4, 8)  # joblib parallelism for PELT fitting
