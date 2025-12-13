from pathlib import Path
from typing import Callable, Any
import os

import numpy as np

import mlflow
import perun
from perun.data_model.data import DataNode, MetricType
from perun.processing import processDataNode

import torch.distributed as dist


_PERUN_MLFLOW_CALLBACK_REGISTERED = False

def log_perun_metrics_to_mlflow(root: DataNode) -> None:
    cfg = getattr(perun, "config", None)
    processed_root = processDataNode(root, cfg, force_process=False) if cfg is not None else root

    run = mlflow.active_run()
    if run is None:
        return

    if not _is_main_process():
        return

    def find_first_metric(node: DataNode, metric_type: MetricType):
        metrics = getattr(node, "metrics", None)
        if metrics and metric_type in metrics:
            return float(metrics[metric_type].value)

        for child in getattr(node, "nodes", {}).values():
            val = find_first_metric(child, metric_type)
            if val is not None:
                return val
        return None

    def fraction_zero_samples_for_metric(node: DataNode, metric_type: MetricType):
        """Traverse DataNode tree and compute fraction of samples == 0
        for the given metric_type, based on raw_data.values.
        """
        zero_samples = 0
        total_samples = 0

        def traverse(n: DataNode):
            nonlocal zero_samples, total_samples
            metrics = getattr(n, "metrics", None)
            raw = getattr(n, "raw_data", None)

            if raw is not None and metrics and metric_type in metrics:
                values = np.asarray(raw.values).ravel()
                if values.size:
                    values = values[~np.isnan(values)]
                if values.size:
                    zero_samples += np.count_nonzero(values == 0)
                    total_samples += values.size

            for child in getattr(n, "nodes", {}).values():
                traverse(child)

        traverse(node)

        if total_samples == 0:
            return None
        return zero_samples / total_samples

    # -------- existing metrics --------
    total_energy_j = find_first_metric(processed_root, MetricType.ENERGY)
    runtime_s      = find_first_metric(processed_root, MetricType.RUNTIME)
    co2_kg         = find_first_metric(processed_root, MetricType.CO2)
    money          = find_first_metric(processed_root, MetricType.MONEY)

    # -------- GPU aggregate metrics --------
    gpu_power_w      = find_first_metric(processed_root, MetricType.GPU_POWER)
    gpu_util_percent = find_first_metric(processed_root, MetricType.GPU_UTIL)
    gpu_mem_bytes    = find_first_metric(processed_root, MetricType.GPU_MEM)

    # -------- GPU idle / mem zero fractions --------
    gpu_idle_fraction = fraction_zero_samples_for_metric(processed_root, MetricType.GPU_UTIL)
    gpu_mem_zero_fraction = fraction_zero_samples_for_metric(processed_root, MetricType.GPU_MEM)

    def log_if_not_none(name: str, value):
        if value is not None:
            mlflow.log_metric(name, float(value))

    # Overall logs
    log_if_not_none("perun_energy_joules", total_energy_j)
    log_if_not_none("perun_co2_kg", co2_kg)
    log_if_not_none("perun_cost", money)

    if total_energy_j is not None:
        energy_kwh = total_energy_j / 3.6e6
        log_if_not_none("perun_energy_kwh", energy_kwh)

    if total_energy_j is not None and runtime_s is not None and runtime_s > 0:
        avg_power_w = total_energy_j / runtime_s
        log_if_not_none("perun_avg_power_watts", avg_power_w)

    # GPU logs
    log_if_not_none("perun_gpu_power_watts", gpu_power_w)
    log_if_not_none("perun_gpu_util_percent", gpu_util_percent)

    if gpu_mem_bytes is not None:
        gpu_mem_gb = gpu_mem_bytes / (1024 ** 3)
        log_if_not_none("perun_gpu_mem_gb", gpu_mem_gb)

    # GPU idle % (util == 0)
    if gpu_idle_fraction is not None:
        log_if_not_none("perun_gpu_idle_fraction", gpu_idle_fraction)
        log_if_not_none("perun_gpu_idle_percent", gpu_idle_fraction * 100.0)

    # GPU mem “unused” % (memory == 0)
    if gpu_mem_zero_fraction is not None:
        log_if_not_none("perun_gpu_mem_zero_fraction", gpu_mem_zero_fraction)
        log_if_not_none("perun_gpu_mem_zero_percent", gpu_mem_zero_fraction * 100.0)

def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    try:
        return dist.get_rank() == 0
    except RuntimeError:
        return True

def _ensure_perun_callback_registered():
    global _PERUN_MLFLOW_CALLBACK_REGISTERED
    if not _PERUN_MLFLOW_CALLBACK_REGISTERED:
        perun.register_callback(log_perun_metrics_to_mlflow)
        _PERUN_MLFLOW_CALLBACK_REGISTERED = True

def run_with_perun(
    fn: Callable[..., Any],
    *args,
    data_out: str,
    format: str = "json",
    **kwargs,
) -> Any:
    """
    Run `fn(*args, **kwargs)` with Perun energy tracking and log to the
    currently-active MLflow run (if any).
    """

    @perun.perun(data_out=data_out, format=format)
    def _wrapped():
        _ensure_perun_callback_registered()
        return fn(*args, **kwargs)

    return _wrapped()