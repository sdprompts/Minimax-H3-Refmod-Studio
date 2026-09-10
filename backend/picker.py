"""Native folder/file dialogs for the local studio (Windows-first)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class PickError(RuntimeError):
    pass


def _initial_dir(initial: str) -> str:
    raw = (initial or "").strip().strip('"')
    if not raw:
        return str(Path.home())
    path = Path(raw).expanduser()
    if path.is_file():
        path = path.parent
    if path.is_dir():
        return str(path)
    return str(Path.home())


def _run(args: list[str], script: str | None = None) -> str:
    kwargs: dict = {
        "args": args,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 600,
    }
    if script is not None:
        kwargs["input"] = script
    try:
        proc = subprocess.run(**kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PickError(str(exc)) from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise PickError(err or f"picker exited {proc.returncode}")
    return (proc.stdout or "").strip()


def _tk_folder(title: str, initial: str) -> str:
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', 1)\n"
        f"p = filedialog.askdirectory(title={title!r}, initialdir={initial!r}) or ''\n"
        "print(p)\n"
        "root.destroy()\n"
    )
    return _run([sys.executable, "-"], script)


def _tk_file(title: str, initial: str, filetypes: list[tuple[str, str]]) -> str:
    types = filetypes or [("All files", "*.*")]
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', 1)\n"
        f"p = filedialog.askopenfilename(title={title!r}, initialdir={initial!r}, filetypes={types!r}) or ''\n"
        "print(p)\n"
        "root.destroy()\n"
    )
    return _run([sys.executable, "-"], script)


def _ps_folder(title: str, initial: str) -> str:
    script = f"""
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = {title!r}
$d.SelectedPath = {initial!r}
$d.ShowNewFolderButton = $true
[void][System.Windows.Forms.Application]::EnableVisualStyles()
$r = $d.ShowDialog()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {{ $d.SelectedPath }}
"""
    return _run(["powershell", "-NoProfile", "-STA", "-Command", script])


def _ps_file(title: str, initial: str, filter_text: str) -> str:
    script = f"""
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = {title!r}
$d.InitialDirectory = {initial!r}
$d.Filter = {filter_text!r}
$d.CheckFileExists = $true
[void][System.Windows.Forms.Application]::EnableVisualStyles()
$r = $d.ShowDialog()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {{ $d.FileName }}
"""
    return _run(["powershell", "-NoProfile", "-STA", "-Command", script])


def _filter_text(filetypes: list[tuple[str, str]] | None) -> str:
    pairs = filetypes or [("All files", "*.*")]
    return "|".join(f"{name} ({pat})|{pat}" for name, pat in pairs)


def pick_folder(title: str = "Select folder", initial: str = "") -> str:
    start = _initial_dir(initial)
    errors: list[str] = []
    for fn in (_tk_folder, _ps_folder):
        try:
            return fn(title or "Select folder", start)
        except PickError as exc:
            errors.append(str(exc))
    raise PickError("Could not open a folder picker: " + " | ".join(errors))


def pick_file(
    title: str = "Select file",
    initial: str = "",
    filetypes: list[tuple[str, str]] | None = None,
) -> str:
    start = _initial_dir(initial)
    errors: list[str] = []
    try:
        return _tk_file(title or "Select file", start, filetypes or [("All files", "*.*")])
    except PickError as exc:
        errors.append(str(exc))
    if os.name == "nt":
        try:
            return _ps_file(title or "Select file", start, _filter_text(filetypes))
        except PickError as exc:
            errors.append(str(exc))
    raise PickError("Could not open a file picker: " + " | ".join(errors))
