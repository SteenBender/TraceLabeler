Idea inspired by [mdreisler](https://github.com/mdreisler)

# Active-Learning Trace Labeler

Interactive GUI for labeling multi-channel microscopy traces using active learning.
Built with tkinter and matplotlib.

## Quick Start

### 1. Install dependencies

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate gui-labeler

# Option B: pip
pip install -r requirements.txt
```

### 2. Prepare data (run on server or locally)

```bash
python -m gui_labeler --prepare --data-root /path/to/data
```

This loads `.mat` files from each experiment subdirectory (`ExNN/`), runs PELT
changepoint detection on every channel, and saves everything to
`results/trace_data.pkl`.

### 3. Launch the GUI

```bash
python -m gui_labeler
```

If a pre-computed `results/trace_data.pkl` exists in the working directory it
will be loaded automatically. Otherwise the tool falls back to computing from
`.mat` files directly (slower).

## Usage

1. **Add label classes** using the sidebar.
2. **Label traces** — click *Show Random Traces* and assign labels.
3. **Train the model** — click *Train Model* to fit a classifier on your labels.
4. **Review predictions** — select a label and use *Show Top* / *Show Low* to
   review high-confidence and uncertain predictions.
5. **Save labels** — labels are saved as JSON to `labels/labels.json`.

### PELT Tuner

Click *PELT Tuner* to interactively adjust changepoint detection parameters
(`pen_mult`, `min_plateau`) on a sample of traces, then apply to all tracks.

### Patch Viewer

Click *Show Patches* to view animated TIFF crops for the currently displayed
traces (requires raw TIFF files in the data root).

## Configuration

| CLI flag       | Env variable | Default                  | Description                            |
|---------------|--------------|--------------------------|----------------------------------------|
| `--data-root` | `DATA_ROOT`  | (set in `config.py`)     | Root dir containing `ExNN/` folders    |
| `--pkl`       |              | `results/trace_data.pkl` | Pre-computed data pickle               |
| `--labels`    |              | `labels/labels.json`     | Labels JSON path                       |
|               | `TIFF_ROOT`  | same as `DATA_ROOT`      | Root for raw TIFF files (patch viewer) |

Edit `gui_labeler/config.py` to set experiment names, channel names, channel
colours, and single-molecule calibration constants for your dataset.

## Data Layout

```
data_root/
├── Ex01/
│   ├── ProcessedTracks.mat
│   ├── steps_<channel>.mat   # one per channel
│   ├── C1-*.tif              # channel 1 TIFFs (optional, for patch viewer)
│   ├── C2-*.tif              # channel 2
│   └── C3-*.tif              # channel 3
├── Ex02/
│   └── ...
└── ExNN/
    └── ...
```

## Project Structure

```
├── gui_labeler/          # main package
│   ├── config.py         # paths, channels, calibration
│   ├── labeler.py        # active-learning core
│   ├── features.py       # feature extraction
│   ├── pelt.py           # PELT changepoint detection
│   ├── serialization.py  # pickle save/load helpers
│   └── gui/              # tkinter GUI
├── notebooks/            # analysis notebooks
├── labels/               # saved label JSON files
└── results/              # outputs (pkl, csv)
```
