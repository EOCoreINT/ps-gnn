"""
ps_gnn.data.fetch_benchmarks
=============================

Utilities for fetching and organizing the benchmark InSAR datasets used to
train and evaluate ``ps-gnn``.

This module is a thin orchestration layer over ``pygeofetch`` (the sibling
package this project will eventually plug into). Because ``pygeofetch`` is
not a hard dependency of ``ps-gnn`` (the package must remain usable
standalone), we attempt to import it lazily and fall back to a
:class:`_MockPygeofetchClient` that mimics its public API and generates
small, deterministic synthetic stacks. This keeps the rest of the codebase
(and CI) fully functional without requiring credentials for a real data
provider (e.g. ASF, Copernicus Data Space) or multi-gigabyte downloads.

Benchmark sites
----------------
The five reference sites used throughout ``ps-gnn``'s development,
representing a spread of PS scenarios (dense urban, volcanic deformation,
post-seismic subsidence, and rapid urbanization):

============================  =================  ==========================
Site                          Scenes             Notes
============================  =================  ==========================
Mexico City                   67                 Extreme aquifer subsidence
Amatrice                      N/A (event-based)  Co-seismic deformation
Piton de la Fournaise         N/A (event-based)  Volcanic inflation/deflation
Berlin                        N/A                Dense urban, low deformation
Jakarta                       N/A                Rapid subsidence, land cover mix
============================  =================  ==========================
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

BenchmarkSite = Literal[
    "mexico_city",
    "amatrice",
    "piton_de_la_fournaise",
    "berlin",
    "jakarta",
]

# Canonical metadata for each benchmark site. ``n_scenes`` follows the
# project brief (Mexico City has a documented 67-scene stack); other sites
# use representative defaults that can be overridden by the caller.
BENCHMARK_REGISTRY: dict[BenchmarkSite, dict] = {
    "mexico_city": {
        "bbox": (-99.30, 19.25, -98.95, 19.55),
        "n_scenes": 67,
        "description": "Extreme groundwater-extraction-driven subsidence.",
    },
    "amatrice": {
        "bbox": (13.25, 42.60, 13.35, 42.70),
        "n_scenes": 24,
        "description": "Co-seismic and post-seismic deformation (2016 earthquake).",
    },
    "piton_de_la_fournaise": {
        "bbox": (55.68, -21.28, 55.80, -21.20),
        "n_scenes": 40,
        "description": "Volcanic edifice inflation/deflation cycles.",
    },
    "berlin": {
        "bbox": (13.30, 52.45, 13.50, 52.60),
        "n_scenes": 55,
        "description": "Dense low-rise/high-rise urban fabric, low background deformation.",
    },
    "jakarta": {
        "bbox": (106.75, -6.25, 106.95, -6.05),
        "n_scenes": 60,
        "description": "Rapid coastal subsidence with heterogeneous land cover.",
    },
}


@dataclass
class BenchmarkPaths:
    """Container for the on-disk paths of a fetched benchmark stack.

    Attributes
    ----------
    site : str
        Benchmark site identifier (see :data:`BENCHMARK_REGISTRY`).
    root : pathlib.Path
        Root directory containing all files for this site.
    amplitude_stack : pathlib.Path
        Path to the multi-temporal amplitude stack (GeoTIFF, shape
        ``(n_scenes, H, W)``).
    phase_stack : pathlib.Path
        Path to the multi-temporal wrapped phase stack (GeoTIFF, same shape).
    dem : pathlib.Path
        Path to the co-registered DEM (GeoTIFF, shape ``(H, W)``).
    metadata : pathlib.Path
        Path to a JSON sidecar with acquisition dates, incidence angle, etc.
    """

    site: str
    root: Path
    amplitude_stack: Path
    phase_stack: Path
    dem: Path
    metadata: Path
    extra: dict[str, Path] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        """Return all paths as strings, convenient for JSON serialization."""
        out = {
            "site": self.site,
            "root": str(self.root),
            "amplitude_stack": str(self.amplitude_stack),
            "phase_stack": str(self.phase_stack),
            "dem": str(self.dem),
            "metadata": str(self.metadata),
        }
        out.update({k: str(v) for k, v in self.extra.items()})
        return out


class _MockPygeofetchClient:
    """Fallback client used when the real ``pygeofetch`` package is absent.

    Generates small, deterministic synthetic GeoTIFF-like stacks (as raw
    ``.npy`` files with a JSON sidecar describing georeferencing, to avoid
    a hard dependency on ``rasterio`` at fetch time) so that downstream
    modules and tests can run end-to-end without network access.

    Notes
    -----
    This mock is intentionally lightweight — it exists purely to unblock
    local development and CI, not to produce scientifically meaningful
    data. Real usage should install ``pygeofetch`` and its credentials.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def download_stack(
        self,
        site: BenchmarkSite,
        target_dir: Path,
        n_scenes: int,
        tile_size: int = 128,
    ) -> BenchmarkPaths:
        target_dir.mkdir(parents=True, exist_ok=True)

        amp = self._rng.gamma(shape=4.0, scale=25.0, size=(n_scenes, tile_size, tile_size))
        phase = self._rng.uniform(-np.pi, np.pi, size=(n_scenes, tile_size, tile_size))
        dem = 200.0 + 50.0 * self._rng.standard_normal((tile_size, tile_size))

        amp_path = target_dir / "amplitude_stack.npy"
        phase_path = target_dir / "phase_stack.npy"
        dem_path = target_dir / "dem.npy"
        meta_path = target_dir / "metadata.json"

        np.save(amp_path, amp.astype(np.float32))
        np.save(phase_path, phase.astype(np.float32))
        np.save(dem_path, dem.astype(np.float32))

        meta = {
            "site": site,
            "n_scenes": n_scenes,
            "tile_size": tile_size,
            "acquisition_dates": [f"2020-{(i % 12) + 1:02d}-01" for i in range(n_scenes)],
            "incidence_angle_deg": float(np.round(self._rng.uniform(30, 45), 2)),
            "bbox": BENCHMARK_REGISTRY[site]["bbox"],
            "mock_data": True,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return BenchmarkPaths(
            site=site,
            root=target_dir,
            amplitude_stack=amp_path,
            phase_stack=phase_path,
            dem=dem_path,
            metadata=meta_path,
        )


def _get_pygeofetch_client():
    """Attempt to import and instantiate the real ``pygeofetch`` client.

    Returns
    -------
    object or None
        An instance exposing a ``download_stack`` method compatible with
        :class:`_MockPygeofetchClient`, or ``None`` if ``pygeofetch`` is
        not installed.
    """
    try:
        import pygeofetch  # type: ignore[import-not-found]

        return pygeofetch.InSARClient()  # type: ignore[attr-defined]
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 -- pragma: no cover - defensive fallback to mock client
        logger.warning("Found pygeofetch but failed to initialize client: %s", exc)
        return None


def fetch_benchmark_dataset(
    site: BenchmarkSite,
    target_dir: str | Path,
    n_scenes: int | None = None,
    force: bool = False,
) -> BenchmarkPaths:
    """Fetch (or reuse cached) benchmark InSAR data for a given site.

    Parameters
    ----------
    site : {"mexico_city", "amatrice", "piton_de_la_fournaise", "berlin", "jakarta"}
        Benchmark site identifier.
    target_dir : str or pathlib.Path
        Directory in which the stack will be stored / cached.
    n_scenes : int, optional
        Number of SAR acquisitions to fetch. Defaults to the registry value
        for the site (e.g. 67 for Mexico City).
    force : bool, default False
        If True, re-download / regenerate even if cached files exist.

    Returns
    -------
    BenchmarkPaths
        Paths to the fetched amplitude/phase stacks, DEM, and metadata.

    Raises
    ------
    ValueError
        If ``site`` is not a recognized benchmark identifier.
    """
    if site not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark site '{site}'. Valid options: {sorted(BENCHMARK_REGISTRY)}"
        )

    target_dir = Path(target_dir)
    n_scenes = n_scenes or BENCHMARK_REGISTRY[site]["n_scenes"]

    metadata_path = target_dir / "metadata.json"
    if metadata_path.exists() and not force:
        logger.info("Reusing cached benchmark data for '%s' at %s", site, target_dir)
        return BenchmarkPaths(
            site=site,
            root=target_dir,
            amplitude_stack=target_dir / "amplitude_stack.npy",
            phase_stack=target_dir / "phase_stack.npy",
            dem=target_dir / "dem.npy",
            metadata=metadata_path,
        )

    client = _get_pygeofetch_client()
    if client is None:
        warnings.warn(
            "pygeofetch is not installed; falling back to a synthetic mock dataset. "
            "Install pygeofetch for real Sentinel-1 stacks: `pip install pygeofetch`.",
            stacklevel=2,
        )
        client = _MockPygeofetchClient()

    logger.info(
        "Fetching benchmark '%s' (%d scenes) into %s using %s",
        site,
        n_scenes,
        target_dir,
        type(client).__name__,
    )
    return client.download_stack(site=site, target_dir=target_dir, n_scenes=n_scenes)


def fetch_all_benchmarks(
    base_dir: str | Path, force: bool = False
) -> dict[BenchmarkSite, BenchmarkPaths]:
    """Fetch every registered benchmark site under a common base directory.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Parent directory; each site is stored in ``base_dir / site``.
    force : bool, default False
        If True, re-download / regenerate every site even if cached.

    Returns
    -------
    dict[str, BenchmarkPaths]
        Mapping from site name to its fetched paths.
    """
    base_dir = Path(base_dir)
    results: dict[BenchmarkSite, BenchmarkPaths] = {}
    for site in BENCHMARK_REGISTRY:
        results[site] = fetch_benchmark_dataset(site, base_dir / site, force=force)
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    paths = fetch_benchmark_dataset("mexico_city", target_dir="data/mexico_city")
    print(json.dumps(paths.as_dict(), indent=2))
