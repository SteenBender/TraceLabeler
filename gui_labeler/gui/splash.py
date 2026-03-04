"""Loading splash window that captures print output during startup."""

import sys
import tkinter as tk
from tkinter import ttk


class _TextRedirector:
    """Tee stdout to both the original stream and a tkinter Text widget."""

    def __init__(self, text_widget, original):
        self._widget = text_widget
        self._original = original

    def write(self, s):
        self._original.write(s)
        try:
            self._widget.configure(state=tk.NORMAL)
            self._widget.insert(tk.END, s)
            self._widget.see(tk.END)
            self._widget.configure(state=tk.DISABLED)
            # Full update() so the window repaints (including progress bar)
            # even while the main thread is blocked in long computations.
            self._widget.winfo_toplevel().update()
        except tk.TclError:
            pass  # window already destroyed

    def flush(self):
        self._original.flush()


class LoadingSplash:
    """A small window that shows print output while data loads.

    Usage::

        root = tk.Tk(); root.withdraw()
        splash = LoadingSplash(root, "Loading data …")
        # ... any code that prints ...
        splash.close()
        root.deiconify()
    """

    def __init__(self, parent, title="Loading …"):
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("620x360")
        self.win.resizable(True, True)

        ttk.Label(self.win, text=title, font=("Helvetica", 13, "bold"),
                  padding=(10, 8)).pack(anchor=tk.W)

        self._text = tk.Text(
            self.win, wrap=tk.WORD, state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4", font=("Menlo", 11),
            relief=tk.FLAT, padx=8, pady=6,
        )
        self._text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self._progress = ttk.Progressbar(self.win, mode="indeterminate")
        self._progress.pack(fill=tk.X, padx=6, pady=(0, 8))
        self._progress.start(15)

        self.win.update()

        # Redirect stdout
        self._orig_stdout = sys.stdout
        sys.stdout = _TextRedirector(self._text, self._orig_stdout)

    def close(self):
        """Restore stdout and destroy the splash window."""
        sys.stdout = self._orig_stdout
        self._progress.stop()
        try:
            self.win.destroy()
        except tk.TclError:
            pass
