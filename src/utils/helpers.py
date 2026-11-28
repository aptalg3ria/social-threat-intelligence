from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | os.PathLike) -> str:
    """
    Resolve a path reliably across operating systems and working directories.

    - Absolute paths are returned as normalized strings.
    - Relative paths are checked against PROJECT_ROOT first, then the current
      working directory. If neither exists yet, returns the PROJECT_ROOT-joined
      path so callers can either create it or report a deterministic path.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)

    proj_candidate = PROJECT_ROOT / p
    if proj_candidate.exists():
        return str(proj_candidate)

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str(proj_candidate)


def ensure_numpy_pickle_compat() -> None:
    """
    Register `numpy._core` module aliases on NumPy 1.x environments so joblib
    artifacts serialized under NumPy 2.x load cleanly across machines.
    """
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
        for sub in ("multiarray", "umath", "numeric", "_multiarray_umath", "fromnumeric"):
            mod = getattr(np.core, sub, None)
            if mod is not None:
                sys.modules[f"numpy._core.{sub}"] = mod


def save_model(model, path: str | os.PathLike) -> None:
    resolved = str(Path(path))
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump(model, resolved)
    print(f"Model saved successfully: {resolved}")


def load_model(path: str | os.PathLike):
    resolved = resolve_project_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Model not found: {path}")

    ensure_numpy_pickle_compat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(resolved)
