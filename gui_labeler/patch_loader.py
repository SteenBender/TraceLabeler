"""On-demand TIFF patch loading for the microscopy patch viewer."""

import glob
import threading

import numpy as np

from . import config

# (exp, ch, track_idx) -> uint16 ndarray (T_track, 2*HALF, 2*HALF), or None on failure
_patch_cache: dict = {}
_patch_cache_lock = threading.Lock()

# C-index (1-based) for each channel, matching the Cx-*.tif filename prefix
_CHANNEL_C_IDX = {"clathrin": 1, "hsc70": 2, "auxilin": 3}


def _find_channel_tiff(exp, ch):
    ci = _CHANNEL_C_IDX[ch]
    matches = sorted(glob.glob(str(config.TIFF_ROOT / exp / f"C{ci}-*.tif")))
    return matches[0] if matches else None


def _load_patches_for_page(candidates, labeler, half=5):
    """Read only the frames used by each track on the current page.

    Groups candidates by experiment so each TIFF file is opened at most once.
    Results are stored in _patch_cache[(exp, ch, idx)].
    """
    import tifffile

    # Group by experiment, skip already-cached tracks
    by_exp: dict = {}
    with _patch_cache_lock:
        for cand in candidates:
            exp, idx = cand["exp"], cand["idx"]
            if any((exp, ch, idx) not in _patch_cache for ch in config.CHANNELS):
                by_exp.setdefault(exp, []).append(cand)

    for exp, exp_cands in by_exp.items():
        # Collect per-track info and the union of all needed frame indices
        track_info = {}  # idx -> (cx, cy, frame_arr)
        all_frames: set = set()
        for cand in exp_cands:
            idx = cand["idx"]
            tr = labeler.all_data[exp][idx]
            med_x = np.nanmedian(tr["x"])
            med_y = np.nanmedian(tr["y"])
            if np.isnan(med_x) or np.isnan(med_y):
                # Skip tracks with no valid coordinates
                with _patch_cache_lock:
                    for ch in config.CHANNELS:
                        _patch_cache[(exp, ch, idx)] = None
                continue
            cx = int(round(float(med_x)))
            cy = int(round(float(med_y)))
            frame_arr = np.asarray(tr["frames"], dtype=int)
            track_info[idx] = (cx, cy, frame_arr)
            all_frames.update(frame_arr.tolist())

        if not all_frames:
            continue

        sorted_frames = sorted(all_frames)
        frame_to_pos = {f: i for i, f in enumerate(sorted_frames)}

        for ch in config.CHANNELS:
            with _patch_cache_lock:
                uncached = [c for c in exp_cands if (c["exp"], ch, c["idx"]) not in _patch_cache]
            if not uncached:
                continue
            path = _find_channel_tiff(exp, ch)
            if path is None:
                with _patch_cache_lock:
                    for c in uncached:
                        _patch_cache[(exp, ch, c["idx"])] = None
                continue
            try:
                with tifffile.TiffFile(path) as tif:
                    T = len(tif.pages)
                    # Read only the union of needed frames in one pass
                    fidx = np.clip(np.array(sorted_frames, dtype=int) - 1, 0, T - 1)
                    frames_data = np.stack(
                        [tif.pages[int(fi)].asarray() for fi in fidx]
                    )  # (N_unique_frames, H, W)
                H, W = frames_data.shape[1], frames_data.shape[2]
                with _patch_cache_lock:
                    for c in uncached:
                        idx = c["idx"]
                        if idx not in track_info:
                            # Track was skipped (NaN coords) — already cached as None
                            continue
                        cx, cy, frame_arr = track_info[idx]
                        y0, y1 = max(0, cy - half), min(H, cy + half)
                        x0, x1 = max(0, cx - half), min(W, cx + half)
                        pos = [frame_to_pos[f] for f in frame_arr.tolist()]
                        sub = frames_data[pos, y0:y1, x0:x1]
                        # Pad to full (2*half, 2*half) when near image border
                        out = np.zeros((len(frame_arr), 2 * half, 2 * half), dtype=np.uint16)
                        oy, ox = half - (cy - y0), half - (cx - x0)
                        out[:, oy : oy + (y1 - y0), ox : ox + (x1 - x0)] = sub
                        _patch_cache[(exp, ch, idx)] = out
            except Exception as exc:
                print(f"[PatchViewer] {exp}/{ch}: {exc}")
                with _patch_cache_lock:
                    for c in uncached:
                        _patch_cache[(exp, ch, c["idx"])] = None
