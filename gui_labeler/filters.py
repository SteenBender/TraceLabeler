"""Trace filters — mark traces as 'valid' or 'ignored' based on PELT step properties."""

import numpy as np

from .features import get_pelt_steps

# Sub-label values
VALID = "valid"
IGNORED = "ignored"

# Filter names — used as main labels for ignored traces
FILTER_EXISTENCE = "existence"
FILTER_INTENSITY = "intensity"
FILTER_TEMPORAL = "temporal"


def _first_nonnoise_step_frame(nr, ch, noise_thresh):
    """Return the frame index of the first step whose absolute size exceeds
    *noise_thresh*, or np.inf if no such step exists."""
    frames, sizes = get_pelt_steps(nr, ch)
    for frame, size in zip(frames, sizes):
        if abs(size) > noise_thresh:
            return int(frame)
    return np.inf


def apply_existence_filter(tr, nr, channels, min_steps, step_direction=None):
    """Existence & Recruitment Check.

    Parameters
    ----------
    step_direction : dict, optional
        Maps channel name → ``"positive"``, ``"negative"``, or ``"both"``.
        Defaults to ``"positive"`` for channels not listed.

    Returns
    -------
    bool
        True if the trace passes (is valid), False if it should be ignored.
    """
    step_direction = step_direction or {}
    for ch in channels:
        if ch not in min_steps:
            continue
        required = min_steps[ch]
        _, sizes = get_pelt_steps(nr, ch)
        sig_range = tr[ch].max() - tr[ch].min() + 1e-6
        thresh = 0.05 * sig_range
        direction = step_direction.get(ch, "positive")
        if direction == "negative":
            n_sig = int(np.sum(sizes < -thresh))
        elif direction == "both":
            n_sig = int(np.sum(np.abs(sizes) > thresh))
        else:  # "positive" (default)
            n_sig = int(np.sum(sizes > thresh))
        if n_sig < required:
            return False
    return True


def apply_intensity_filter(tr, nr, channels, max_step_size):
    """Intensity Stability filter.

    Returns
    -------
    bool
        True if the trace passes (is valid), False if it should be ignored.
    """
    for ch in channels:
        if ch not in max_step_size:
            continue
        limit = max_step_size[ch]
        _, sizes = get_pelt_steps(nr, ch)
        if len(sizes) > 0 and np.max(np.abs(sizes)) > limit:
            return False
    return True


def apply_temporal_order_filter(tr, nr, channel_order, noise_thresh, leniency=0.0):
    """Temporal Order filter.

    Returns
    -------
    bool
        True if the trace passes (is valid), False if it should be ignored.
    """
    if len(channel_order) < 2:
        return True

    first_frames = {}
    for ch in channel_order:
        thresh = noise_thresh.get(ch, 0.0)
        first_frames[ch] = _first_nonnoise_step_frame(nr, ch, thresh)

    for i in range(len(channel_order) - 1):
        ch_a = channel_order[i]
        ch_b = channel_order[i + 1]
        fa = first_frames[ch_a]
        fb = first_frames[ch_b]
        if fb < fa - leniency:
            return False
    return True


def apply_all_filters(tr, nr, filter_config):
    """Run all enabled filters on a single trace.

    Returns
    -------
    tuple[str, str | None]
        ``(sub_label, reason)`` where *sub_label* is ``VALID`` or ``IGNORED``
        and *reason* is the "+"-joined names of failing filters (e.g.
        ``"existence"``, ``"intensity+temporal"``), or ``None`` when valid.
    """
    failing = []

    ex = filter_config.get("existence", {})
    if ex.get("enabled", False):
        if not apply_existence_filter(
            tr, nr, ex.get("channels", []), ex.get("min_steps", {}),
            ex.get("step_direction", {})
        ):
            failing.append(FILTER_EXISTENCE)

    inten = filter_config.get("intensity", {})
    if inten.get("enabled", False):
        if not apply_intensity_filter(
            tr, nr, inten.get("channels", []), inten.get("max_step_size", {})
        ):
            failing.append(FILTER_INTENSITY)

    temp = filter_config.get("temporal", {})
    if temp.get("enabled", False):
        if not apply_temporal_order_filter(
            tr,
            nr,
            temp.get("channel_order", []),
            temp.get("noise_thresh", {}),
            temp.get("leniency", 0.0),
        ):
            failing.append(FILTER_TEMPORAL)

    if failing:
        return IGNORED, "+".join(failing)
    return VALID, None
