from __future__ import annotations

import os
import random

from research_navigator.common.seeds import DEFAULT_SEED, seed_everything


def test_returns_the_seed_used() -> None:
    assert seed_everything(42) == 42


def test_default_seed_is_applied() -> None:
    assert seed_everything() == DEFAULT_SEED


def test_python_random_is_reproducible_after_seeding() -> None:
    seed_everything(7)
    first = [random.random() for _ in range(5)]
    seed_everything(7)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_different_seeds_produce_different_streams() -> None:
    seed_everything(1)
    a = [random.random() for _ in range(5)]
    seed_everything(2)
    b = [random.random() for _ in range(5)]
    assert a != b


def test_pythonhashseed_env_is_set() -> None:
    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_numpy_global_rng_is_seeded_when_available() -> None:
    # numpy is present (transitive dep of onnxruntime/FastEmbed); assert it is actually
    # pinned so two seeded draws match. If numpy were absent this import would be skipped
    # by _seed_numpy and this test is simply not meaningful — but it is present here.
    import numpy as np

    seed_everything(99)
    a = np.random.rand(4)
    seed_everything(99)
    b = np.random.rand(4)
    assert (a == b).all()
