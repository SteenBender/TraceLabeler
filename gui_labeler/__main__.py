"""Entry point: python -m gui_labeler [--prepare]"""

import argparse
import os
from pathlib import Path

from . import config
from .serialization import save_prepared_data


def main():
    parser = argparse.ArgumentParser(
        description="Active-Learning Trace Labeler for sCCV uncoating traces"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Load .mat data, run PELT, save to .pkl (no GUI)",
    )
    parser.add_argument(
        "--data-root",
        default=str("Select data root folder".format(config.DATA_ROOT)),
        help=f"Root directory containing ExNN folders (default: {config.DATA_ROOT})",
    )
    parser.add_argument(
        "--pkl",
        default=config.PICKLE_PATH,
        help=f"Path to pre-computed .pkl file (default: {config.PICKLE_PATH})",
    )
    parser.add_argument(
        "--labels",
        default=config.SAVE_PATH,
        help=f"Path to labels JSON (default: {config.SAVE_PATH})",
    )
    parser.add_argument(
        "--features",
        choices=["pelt", "pretrained"],
        default="pelt",
        help="Feature mode: 'pelt' (hand-crafted PELT features) or 'pretrained' (CNN encoder embeddings)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Active-Learning Trace Labeler")
    print("=" * 60)

    # Apply --data-root override (modifies the shared config module)
    config.DATA_ROOT = Path(args.data_root)

    # ── Prepare mode (server, no display) ──
    if args.prepare:
        if not config.DATA_ROOT.is_dir():
            print(
                f"Error: data root '{config.DATA_ROOT}' does not exist.\n"
                "Set --data-root to the directory containing ExNN folders."
            )
            raise SystemExit(1)
        os.makedirs(os.path.dirname(os.path.abspath(args.pkl)) or ".", exist_ok=True)
        save_prepared_data(args.pkl)
        return

    # ── GUI mode ── the app owns the full startup flow (folder browse,
    # channel config, data loading).  Import tkinter only here so that
    # --prepare works headless.
    import tkinter as tk
    from .gui.app import LabelerApp

    app = LabelerApp(
        pkl_path=args.pkl, labels_path=args.labels, feature_mode=args.features
    )
    app.mainloop()


if __name__ == "__main__":
    main()
