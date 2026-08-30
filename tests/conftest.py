"""
tests.conftest
================

Shared pytest fixtures for the ``ps-gnn`` test suite:

- Deterministic seeding (``seed_everything``, autouse) so every test runs
  with fixed NumPy/PyTorch RNG state, per the reproducibility requirement.
- Small synthetic amplitude/phase/DEM GeoTIFFs (``dummy_geotiffs``) for
  tests that exercise the ``rasterio``-based raster-reading paths in
  :mod:`ps_gnn.data.preprocessing`.
- A tiny in-memory synthetic stack (``tiny_stack``) for tests that only
  need NumPy arrays, avoiding filesystem I/O entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 1234


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed NumPy and (if installed) PyTorch RNGs before every test.

    Autouse so every test in the suite gets deterministic random state
    without needing to remember to call this explicitly.
    """
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(False)  # some GNN scatter ops lack det. kernels
    except ImportError:
        pass
    yield


@pytest.fixture
def tiny_stack():
    """A tiny (T=6, H=10, W=10) synthetic amplitude/phase stack, in memory.

    Returns
    -------
    dict
        ``amplitude``, ``phase`` (T, H, W arrays), ``worldcover`` (H, W),
        ``incidence_angle`` (scalar).
    """
    rng = np.random.default_rng(SEED)
    t_steps, height, width = 6, 10, 10
    amplitude = rng.gamma(shape=4.0, scale=25.0, size=(t_steps, height, width)).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=(t_steps, height, width)).astype(np.float32)
    worldcover = rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 95], size=(height, width))
    return {
        "amplitude": amplitude,
        "phase": phase,
        "worldcover": worldcover,
        "incidence_angle": 38.0,
    }


@pytest.fixture
def dummy_geotiffs(tmp_path, tiny_stack):
    """Write the ``tiny_stack`` fixture out as small real GeoTIFF files.

    Exercises the ``rasterio``-backed raster-loading path in
    :func:`ps_gnn.data.preprocessing._load_raster_stack`, which the
    in-memory ``tiny_stack`` fixture alone does not cover.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's built-in temporary directory fixture.
    tiny_stack : dict
        See :func:`tiny_stack`.

    Returns
    -------
    dict
        Paths to the written ``amplitude.tif``, ``phase.tif``, and
        ``dem.tif`` files, plus the original in-memory arrays for
        cross-checking.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    amplitude = tiny_stack["amplitude"]
    phase = tiny_stack["phase"]
    _t_steps, height, width = amplitude.shape

    transform = from_origin(0, 0, 10, 10)  # 10m pixel spacing, arbitrary origin

    def _write_stack(path, array):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[-2],
            width=array.shape[-1],
            count=array.shape[0] if array.ndim == 3 else 1,
            dtype=array.dtype,
            crs="EPSG:32633",
            transform=transform,
        ) as dst:
            if array.ndim == 3:
                for band in range(array.shape[0]):
                    dst.write(array[band], band + 1)
            else:
                dst.write(array, 1)

    amp_path = tmp_path / "amplitude.tif"
    phase_path = tmp_path / "phase.tif"
    dem_path = tmp_path / "dem.tif"

    rng = np.random.default_rng(SEED)
    dem = (200.0 + 20.0 * rng.standard_normal((height, width))).astype(np.float32)

    _write_stack(amp_path, amplitude)
    _write_stack(phase_path, phase)
    _write_stack(dem_path, dem)

    return {
        "amplitude_path": amp_path,
        "phase_path": phase_path,
        "dem_path": dem_path,
        "amplitude": amplitude,
        "phase": phase,
        "dem": dem,
    }
