"""Entry point: python -m gui_labeler [--prepare]"""

import argparse
import json
import os
from pathlib import Path

from . import config
from .analytical import run_analytical
from .data_loader import _load_and_pelt
from .labeler import ActiveLabeler
from .serialization import load_prepared_data, save_prepared_data


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
        default=str(config.DATA_ROOT),
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

    # ── GUI mode — import tkinter only here so --prepare works headless ──
    import tkinter as tk
    from tkinter import messagebox, filedialog
    from .gui.app import LabelerApp
    from .gui.splash import LoadingSplash

    # Load data — prefer pickle; if missing, ask user whether to prepare
    if os.path.exists(args.pkl):
        splash_root = tk.Tk()
        splash_root.withdraw()
        splash = LoadingSplash(splash_root, "Loading data …")
        all_data, pelt_results, df_triggered = load_prepared_data(args.pkl)
    else:
        # No pickle found — ask user what to do
        ask_root = tk.Tk()
        ask_root.overrideredirect(True)
        ask_root.geometry("0x0+0+0")
        ask_root.update_idletasks()
        ask_root.attributes("-topmost", True)
        ask_root.withdraw()
        ask_root.lift()
        ask_root.focus_force()

        has_data_root = config.DATA_ROOT.is_dir()

        if has_data_root:
            msg = (
                f"No pre-computed data file found at:\n"
                f"  {args.pkl}\n\n"
                f"Raw .mat files were found in:\n"
                f"  {config.DATA_ROOT}\n\n"
                f"Would you like to prepare the data now?\n"
                f"(This runs PELT fitting and may take a few minutes.)"
            )
            do_prepare = messagebox.askyesno("Prepare Data?", msg, parent=ask_root)
        else:
            msg = (
                f"No pre-computed data file found at:\n"
                f"  {args.pkl}\n\n"
                f"Data root '{config.DATA_ROOT}' does not exist.\n\n"
                f"Would you like to select the folder containing\n"
                f"the ExNN/ experiment directories?"
            )
            do_prepare = messagebox.askyesno("Data Not Found", msg, parent=ask_root)
            if do_prepare:
                chosen = filedialog.askdirectory(
                    title="Select folder containing ExNN/ directories",
                    parent=ask_root,
                )
                if not chosen:
                    ask_root.destroy()
                    raise SystemExit(0)
                config.DATA_ROOT = Path(chosen)
                config.TIFF_ROOT = Path(chosen)

        ask_root.destroy()

        if not do_prepare:
            raise SystemExit(0)

        if not config.DATA_ROOT.is_dir():
            print(f"Error: data root '{config.DATA_ROOT}' does not exist.")
            raise SystemExit(1)

        # Show splash and run preparation
        splash_root = tk.Tk()
        splash_root.withdraw()
        splash = LoadingSplash(splash_root, "Preparing data (this may take a few minutes) …")

        all_data, pelt_results = _load_and_pelt()
        print("Running analytical detection …")
        df_triggered = run_analytical(all_data, pelt_results, config.EXPERIMENTS)

        # Save the pickle so next launch is fast
        os.makedirs(os.path.dirname(os.path.abspath(args.pkl)) or ".", exist_ok=True)
        from .serialization import save_bundle
        saved_mb = save_bundle(all_data, pelt_results, df_triggered, args.pkl)
        print(f"Saved prepared data → {args.pkl} ({saved_mb:.1f} MB)")

    print("Building feature matrix …")
    os.makedirs(os.path.dirname(os.path.abspath(args.labels)) or ".", exist_ok=True)
    labeler = ActiveLabeler(
        all_data,
        pelt_results,
        config.CHANNELS,
        config.CHANNEL_COLORS,
        experiments=config.EXPERIMENTS,
        save_path=args.labels,
        calibration=config.CALIBRATION,
    )
    labeler.df_triggered = df_triggered
    labeler.pkl_path = args.pkl if os.path.exists(args.pkl) else None

    if os.path.exists(args.labels):
        try:
            n = labeler.load(args.labels)
            print(f"  Auto-loaded {n} labels from {args.labels}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  Warning: could not load labels from {args.labels}: {exc}")

    print("Launching GUI …")
    splash.close()
    splash_root.destroy()

    # LabelerApp factory creates its own tk.Tk root
    app = LabelerApp(labeler)
    app.mainloop()


if __name__ == "__main__":
    main()
