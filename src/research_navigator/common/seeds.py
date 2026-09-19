"""Reproducibility helpers: seed every source of randomness the pipeline can touch.

The Research Navigator is deterministic *by construction* almost everywhere — chunk IDs
are content-addressed (``chunk/hashing.py``), routing is rules-first (``agents/``),
hybrid fusion (RRF) is order-stable, and the LLM runs at ``temperature=0.0``. The few
places randomness could still leak in are:

* Python's hash-seed-dependent ``set``/``dict`` iteration order,
* the stdlib ``random`` module (unused in output paths today, seeded defensively so a
  future contributor who *adds* a random component inherits the seed automatically), and
* ``numpy`` (pulled in transitively by onnxruntime / FastEmbed).

``seed_everything`` pins all of them from a single config value so a run is reproducible
and so reproducibility can't silently regress. It is wired into the CLI as a Typer
callback (``cli.py``) that runs before every command, driven by
``Settings.repro`` (env ``RN_REPRO__SEED`` / ``RN_REPRO__SEED_ON_STARTUP``).

No silent failure: ``numpy`` is imported defensively and its absence is logged at debug
level, never swallowed as ``except: pass``.
"""

from __future__ import annotations

import os
import random

import structlog

log = structlog.get_logger(__name__)

DEFAULT_SEED = 1234


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed all known RNGs from one integer. Returns the seed used (handy for logging).

    Note on ``PYTHONHASHSEED``: CPython reads it only at interpreter *startup*, so setting
    it here does not re-randomize the already-running process — it governs any child
    process we spawn and documents intent. The pipeline never depends on ``set``/``dict``
    iteration order for its output, so current-process determinism does not rely on it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    _seed_numpy(seed)
    log.debug("seeded", seed=seed)
    return seed


def _seed_numpy(seed: int) -> None:
    """Seed numpy's legacy global RNG if numpy is importable; otherwise log and move on."""
    try:
        import numpy as np
    except ImportError:
        log.debug("numpy not importable; skipping numpy seed (expected in minimal installs)")
        return
    np.random.seed(seed)
