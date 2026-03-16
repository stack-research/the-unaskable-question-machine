"""
Evolved probes — auto-generated follow-ups that drill into detected cracks.

This module auto-imports all .py files in this directory so that
evolved probes register themselves via @register_probe.
"""

import importlib
from pathlib import Path

_dir = Path(__file__).parent

for f in sorted(_dir.glob("evolved_*.py")):
    module_name = f"src.probes.evolved.{f.stem}"
    try:
        importlib.import_module(module_name)
    except Exception:
        pass  # Don't let a bad evolved probe break everything
