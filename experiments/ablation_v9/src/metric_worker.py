"""Keep official metric process pools separate from the loaded training pickle."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import random
import sys

import numpy as np
import torch


def _initialize_worker(seed: int, cpu_threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    # Spawn isolates the large parent, but would also make every official inner
    # pool spawn/import dependencies again. Linux inner pools can safely fork
    # this small CPU-only process, matching the official Linux execution path.
    if sys.platform.startswith("linux"):
        multiprocessing.set_start_method("fork", force=True)


def _worker_info() -> dict:
    from .data import _DATA_CACHE

    return {
        "pid": os.getpid(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cached_datasets": len(_DATA_CACHE),
        "inner_start_method": multiprocessing.get_start_method(),
    }


def _compute(scores: np.ndarray, labels: np.ndarray, sliding_window: int, protocol: dict) -> dict:
    from .benchmark_eval import official_metrics

    return official_metrics(scores, labels, sliding_window, protocol)


class OfficialMetricWorker:
    """Reuse one non-daemonic spawned process; official nested pools stay intact."""

    def __init__(self, protocol: dict):
        self.protocol = dict(protocol)
        self.executor: ProcessPoolExecutor | None = None
        self.info: dict = {}

    def __enter__(self) -> OfficialMetricWorker:
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(int(self.protocol["seed"]), int(self.protocol["cpu_threads"])),
        )
        try:
            self.info = self.executor.submit(_worker_info).result()
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            raise
        return self

    def compute(self, scores: np.ndarray, labels: np.ndarray, sliding_window: int) -> dict:
        if self.executor is None:
            raise RuntimeError("Metric worker is not running")
        return self.executor.submit(_compute, scores, labels, sliding_window, self.protocol).result()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
