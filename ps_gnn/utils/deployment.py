"""
ps_gnn.utils.deployment
==========================

Production deployment helpers for PS-GNN:

1. :func:`export_to_onnx` — converts a trained :class:`PSGNN` to ONNX,
   with optional FP16 quantization for smaller, faster CPU/edge
   inference.
2. :func:`get_cached_model_weights` — downloads model weights from a
   Zenodo DOI on first use and caches them under ``~/.ps_gnn/models/``,
   so subsequent runs (and offline environments) reuse the local copy.
3. :class:`InferenceRuntime` — a thin wrapper that prefers ONNX Runtime
   with GPU execution, falling back to ONNX Runtime CPU, and finally to
   OpenVINO for CPU inference if ONNX Runtime itself is unavailable —
   so a single call site works across heterogeneous deployment targets.

Notes on graph-model ONNX export
----------------------------------
Exporting a graph neural network (with dynamic node/edge counts and
scatter-based message passing, as used by ``GATConv``) to ONNX is
inherently more fragile than exporting a fixed-shape CNN/MLP: not every
``torch_geometric`` operator has a stable ONNX opset mapping, and the
exported graph is generally only valid for the input's *shape family*
(dynamic axes are declared for the node/edge dimensions to mitigate
this, but exotic ops occasionally still require ``torch_geometric``'s own
compilation utilities rather than vanilla ``torch.onnx.export``). This
module exports on a best-effort basis and raises a clearly-labeled
``RuntimeError`` (rather than a cryptic torch traceback) if export fails,
so callers can decide whether to fall back to native PyTorch inference.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ps_gnn.models.ps_gnn import PSGNN

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".ps_gnn" / "models"


@dataclass
class ModelRelease:
    """Metadata for a versioned, Zenodo-hosted model release.

    Attributes
    ----------
    doi : str
        The Zenodo DOI identifying this release, e.g.
        ``"10.5281/zenodo.1234567"``.
    filename : str
        Expected weights filename within the Zenodo record (e.g.
        ``"ps_gnn_v1.pt"``).
    sha256 : str, optional
        Expected SHA-256 checksum of the downloaded file, verified after
        download if provided (recommended for reproducibility/integrity).
    """

    doi: str
    filename: str
    sha256: str | None = None


def _default_zenodo_downloader(release: ModelRelease, destination: Path) -> None:
    """Download a file from Zenodo via its DOI, using the ``requests`` library.

    Parameters
    ----------
    release : ModelRelease
    destination : pathlib.Path
        Local path to write the downloaded file to.

    Raises
    ------
    ImportError
        If ``requests`` is not installed.
    RuntimeError
        If the download fails (network error, non-200 response, or the
        record/file cannot be resolved from the DOI).
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "Downloading model weights from Zenodo requires the `requests` "
            "package: `pip install requests`."
        ) from exc

    record_id = release.doi.rsplit(".", 1)[-1]
    url = f"https://zenodo.org/record/{record_id}/files/{release.filename}"

    logger.info("Downloading %s from %s ...", release.filename, url)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download model weights for DOI {release.doi} from {url}: {exc}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        f.writelines(response.iter_content(chunk_size=1 << 20))


def get_cached_model_weights(
    release: ModelRelease,
    cache_dir: str | Path | None = None,
    downloader: Callable[[ModelRelease, Path], None] = _default_zenodo_downloader,
    force_download: bool = False,
) -> Path:
    """Return a local path to model weights, downloading + caching on first use.

    Parameters
    ----------
    release : ModelRelease
        Which release to fetch.
    cache_dir : str or pathlib.Path, optional
        Cache directory. Defaults to ``~/.ps_gnn/models/``.
    downloader : callable, optional
        ``(release, destination) -> None`` function that performs the
        actual download. Defaults to a Zenodo-based downloader using
        ``requests``; injectable for testing or alternative hosts
        (institutional mirrors, S3, etc.) without a live network call.
    force_download : bool, default False
        Re-download even if a cached copy exists.

    Returns
    -------
    pathlib.Path
        Local path to the cached weights file.

    Raises
    ------
    RuntimeError
        If the download succeeds but fails the expected SHA-256 checksum.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    destination = cache_dir / release.filename

    if destination.exists() and not force_download:
        logger.info("Using cached model weights at %s", destination)
        return destination

    downloader(release, destination)

    if release.sha256 is not None:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != release.sha256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {release.filename}: "
                f"expected {release.sha256}, got {digest}. File removed."
            )

    logger.info("Cached model weights at %s", destination)
    return destination


def export_to_onnx(
    model: PSGNN,
    output_path: str | Path,
    n_nodes: int = 100,
    n_edges: int = 400,
    fp16: bool = True,
    opset_version: int = 17,
) -> Path:
    """Export a trained PSGNN to ONNX, optionally with FP16 quantization.

    Parameters
    ----------
    model : PSGNN
        A trained model, in evaluation mode (this function calls
        ``model.eval()`` regardless).
    output_path : str or pathlib.Path
        Destination ``.onnx`` file.
    n_nodes, n_edges : int, default 100, 400
        Dummy graph size used to trace the export; the node and edge
        dimensions are declared dynamic (``dynamic_axes``), so the
        exported model accepts different sizes at inference time, but a
        reasonably representative size still helps the tracer avoid
        degenerate (e.g. zero-edge) code paths.
    fp16 : bool, default True
        If True, convert the exported model's weights to FP16 after
        export (via ``onnxconverter_common``), roughly halving model size
        and improving CPU/edge inference latency at a small precision
        cost.
    opset_version : int, default 17
        ONNX opset version to target.

    Returns
    -------
    pathlib.Path
        Path to the exported (and possibly FP16-converted) ONNX model.

    Raises
    ------
    ImportError
        If the ``onnx`` package (or, for FP16, ``onnxconverter_common``)
        is not installed.
    RuntimeError
        If the export itself fails — commonly because an operator used
        inside a graph convolution lacks an ONNX mapping for the target
        opset. The original exception is chained for debugging.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "export_to_onnx requires the `onnx` package: `pip install onnx`."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_x = torch.randn(n_nodes, model.config.in_channels)
    dummy_edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)

    class _ONNXWrapper(torch.nn.Module):
        """Wraps PSGNN.forward to return a single Tensor (logits only).

        ONNX export handles a fixed, flat set of Tensor outputs far more
        reliably than PSGNN's normal ``(logits, attention | None)`` tuple
        return, so the attention output — needed for explainability, not
        for serving predictions — is intentionally dropped here.
        """

        def __init__(self, wrapped: PSGNN) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            logits, _ = self.wrapped(x, edge_index, return_attention=False)
            return logits

    export_module = _ONNXWrapper(model)

    try:
        torch.onnx.export(
            export_module,
            (dummy_x, dummy_edge_index),
            str(output_path),
            input_names=["x", "edge_index"],
            output_names=["logits"],
            dynamic_axes={
                "x": {0: "n_nodes"},
                "edge_index": {1: "n_edges"},
                "logits": {0: "n_nodes"},
            },
            opset_version=opset_version,
        )
    except Exception as exc:
        raise RuntimeError(
            "ONNX export failed. This is commonly caused by a graph-convolution "
            "operator (e.g. inside GATConv) lacking a stable ONNX mapping for "
            f"opset {opset_version}. Original error: {exc}"
        ) from exc

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    if fp16:
        try:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)
            onnx.save(onnx_model, str(output_path))
            logger.info("Exported FP16-quantized ONNX model to %s", output_path)
        except ImportError:
            logger.warning(
                "fp16=True but `onnxconverter_common` is not installed; "
                "saved a full-precision (FP32) ONNX model instead. "
                "Install with `pip install onnxconverter-common` for FP16."
            )
    else:
        logger.info("Exported ONNX model to %s", output_path)

    return output_path


class InferenceRuntime:
    """Deployment-agnostic inference wrapper with automatic fallback.

    Attempts, in order: ONNX Runtime with GPU execution (if a CUDA
    execution provider and device are available), ONNX Runtime on CPU,
    and finally OpenVINO on CPU. The first backend that initializes
    successfully is used for all subsequent calls to :meth:`predict`.

    Parameters
    ----------
    onnx_path : str or pathlib.Path
        Path to an exported ``.onnx`` model (see :func:`export_to_onnx`).
    prefer_gpu : bool, default True
        Whether to attempt GPU execution before falling back to CPU.

    Attributes
    ----------
    backend : str
        Which backend ended up active: ``"onnxruntime-gpu"``,
        ``"onnxruntime-cpu"``, or ``"openvino-cpu"``.
    """

    def __init__(self, onnx_path: str | Path, prefer_gpu: bool = True) -> None:
        self.onnx_path = str(onnx_path)
        self.backend: str
        self._session = None

        if prefer_gpu and self._try_init_onnxruntime(use_gpu=True):
            self.backend = "onnxruntime-gpu"
        elif self._try_init_onnxruntime(use_gpu=False):
            self.backend = "onnxruntime-cpu"
        elif self._try_init_openvino():
            self.backend = "openvino-cpu"
        else:
            raise RuntimeError(
                "No usable inference backend found. Install `onnxruntime` "
                "(or `onnxruntime-gpu`) or `openvino` to run exported PS-GNN models."
            )

        logger.info("InferenceRuntime initialized with backend: %s", self.backend)

    def _try_init_onnxruntime(self, use_gpu: bool) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        try:
            session = ort.InferenceSession(self.onnx_path, providers=providers)
            if use_gpu and "CUDAExecutionProvider" not in session.get_providers():
                return False  # silently fell back to CPU inside onnxruntime; let our own fallback chain handle it
            self._session = session
            return True
        except Exception as exc:  # noqa: BLE001
            # pragma: no cover - any backend init failure should fall through to the next backend
            logger.warning("onnxruntime initialization failed (use_gpu=%s): %s", use_gpu, exc)
            return False

    def _try_init_openvino(self) -> bool:
        try:
            from openvino.runtime import Core
        except ImportError:
            return False
        try:
            core = Core()
            ov_model = core.read_model(self.onnx_path)
            self._session = core.compile_model(ov_model, "CPU")
            return True
        except Exception as exc:  # noqa: BLE001
            # pragma: no cover - any backend init failure should fall through to the next backend
            logger.warning("OpenVINO initialization failed: %s", exc)
            return False

    def predict(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Run inference through whichever backend was initialized.

        Parameters
        ----------
        x : np.ndarray, shape (N, 19), dtype float32
        edge_index : np.ndarray, shape (2, E), dtype int64

        Returns
        -------
        np.ndarray, shape (N, n_classes)
            Raw logits.
        """
        if self.backend.startswith("onnxruntime"):
            outputs = self._session.run(
                ["logits"], {"x": x.astype(np.float32), "edge_index": edge_index.astype(np.int64)}
            )
            return outputs[0]
        elif self.backend == "openvino-cpu":
            result = self._session(
                {"x": x.astype(np.float32), "edge_index": edge_index.astype(np.int64)}
            )
            return next(iter(result.values()))
        raise RuntimeError(f"Unknown backend: {self.backend}")  # pragma: no cover
