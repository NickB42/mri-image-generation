from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import mlflow
import mlflow.pytorch

import perun
from perun.data_model.data import MetricType, DataNode
from perun.processing import processDataNode


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Names
    experiment_name: str
    run_name: str

    # Data / paths
    dataset_root: Path
    experiments_root: Path

    # Core hyperparameters
    image_size: int = 128
    batch_size: int = 16
    timesteps: int = 800
    learning_rate: float = 2e-4
    num_epochs: int = 20
    patience: int = 5
    min_delta: float = 1e-4

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True

    # Debug options
    debug_fast: bool = False
    debug_train_steps: int = 10
    debug_val_steps: int = 5

    # Logging
    log_every_n_steps: int = 500


# ------------------------------------------------------------
# Perun → MLflow bridge (generic, reusable)
# ------------------------------------------------------------
def log_perun_metrics_to_mlflow(root: DataNode) -> None:
    cfg = getattr(perun, "config", None)
    processed_root = processDataNode(root, cfg, force_process=False) if cfg is not None else root

    def find_first_metric(node: DataNode, metric_type: MetricType):
        metrics = getattr(node, "metrics", None)
        if metrics and metric_type in metrics:
            return float(metrics[metric_type].value)

        for child in getattr(node, "nodes", {}).values():
            val = find_first_metric(child, metric_type)
            if val is not None:
                return val
        return None

    run = mlflow.active_run()
    if run is None:
        return

    total_energy_j = find_first_metric(processed_root, MetricType.ENERGY)
    runtime_s = find_first_metric(processed_root, MetricType.RUNTIME)
    co2_kg = find_first_metric(processed_root, MetricType.CO2)
    money = find_first_metric(processed_root, MetricType.MONEY)

    def log_if_not_none(name: str, value):
        if value is not None:
            mlflow.log_metric(name, float(value))

    log_if_not_none("perun_energy_joules", total_energy_j)
    log_if_not_none("perun_runtime_seconds", runtime_s)
    log_if_not_none("perun_co2_kg", co2_kg)
    log_if_not_none("perun_cost", money)

    if total_energy_j is not None:
        energy_kwh = total_energy_j / 3.6e6
        log_if_not_none("perun_energy_kwh", energy_kwh)

    if total_energy_j is not None and runtime_s is not None and runtime_s > 0:
        avg_power_w = total_energy_j / runtime_s
        log_if_not_none("perun_avg_power_watts", avg_power_w)


# ------------------------------------------------------------
# Base trainer
# ------------------------------------------------------------
class BaseTrainer(ABC):
    """
    Handles:
      - device setup
      - training / validation loops
      - MLflow logging
      - Perun integration (via helper function below)

    Subclasses must implement:
      - create_dataloaders()
      - build_model()
      - compute_loss(batch, train: bool)
      - optional: extra_params(), get_model_for_logging(), checkpoint_name()
    """

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = self._get_device()

        # Unique run id (from Slurm or timestamp)
        self.run_identifier = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Directories
        self.experiment_dir = cfg.experiments_root / cfg.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.experiment_dir / "models" / self.run_identifier
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.perun_out_dir = self.experiment_dir / "perun_results" / self.run_identifier
        self.perun_out_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.train_loader, self.val_loader = self.create_dataloaders()

        # Model + optimizers
        self.model: nn.Module
        self.build_model()
        self.model.to(self.device)

        self.optimizer, self.scheduler = self.create_optimizers()
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

    # ----------------- hooks to override -----------------

    @abstractmethod
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        ...

    @abstractmethod
    def build_model(self) -> None:
        """Must set self.model."""
        ...

    @abstractmethod
    def compute_loss(self, batch, train: bool) -> torch.Tensor:
        """Return scalar loss for a batch."""
        ...

    def create_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return optimizer, scheduler

    def extra_params(self) -> dict:
        """Override to add model/dataset-specific params to MLflow."""
        return {}

    def get_model_for_logging(self) -> nn.Module:
        """Override if you want to log only a submodule."""
        return self.model

    def checkpoint_name(self) -> str:
        """Filename (without path) for best checkpoint."""
        return f"{self.cfg.run_name}_best.pt"

    # ----------------- core loop -----------------

    def train_loop(self) -> float:
        print("Starting training")
        best_val = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric(
                "learning_rate",
                self.optimizer.param_groups[0]["lr"],
                step=epoch,
            )

            self.scheduler.step(val_loss)

            # early stopping
            if val_loss < best_val - self.cfg.min_delta:
                best_val = val_loss
                epochs_without_improvement = 0
                ckpt_path = self._save_checkpoint()
                print(f"✅ New best val loss: {best_val:.4f} (saved to {ckpt_path})")
            else:
                epochs_without_improvement += 1
                print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= self.cfg.patience:
                print(
                    f"⏹ Early stopping at epoch {epoch} "
                    f"(no val improvement for {self.cfg.patience} epochs).",
                )
                break

        return best_val

    # ----------------- helpers -----------------

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        n_steps = 0

        max_steps = self.cfg.debug_train_steps if self.cfg.debug_fast else None

        for step, batch in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()

            with autocast("cuda", enabled=(self.device.type == "cuda")):
                loss = self.compute_loss(batch, train=True)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            n_steps += 1

            if step % self.cfg.log_every_n_steps == 0:
                avg = running_loss / max(1, n_steps)
                print(f"[epoch {epoch} | step {step}] avg loss: {avg:.4f}")

            if max_steps is not None and step >= max_steps:
                break

        avg_loss = running_loss / max(1, n_steps)
        print(f"Epoch {epoch} | Train loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        running_loss = 0.0
        n_steps = 0

        max_steps = self.cfg.debug_val_steps if self.cfg.debug_fast else None

        for step, batch in enumerate(self.val_loader, start=1):
            loss = self.compute_loss(batch, train=False)
            running_loss += loss.item()
            n_steps += 1

            if max_steps is not None and step >= max_steps:
                break

        avg_loss = running_loss / max(1, n_steps)
        print(f"Epoch {epoch} | Val loss:   {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self) -> Path:
        ckpt_path = self.models_dir / self.checkpoint_name()
        torch.save(self.model.state_dict(), str(ckpt_path))
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
        return ckpt_path

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")
        print("=== PyTorch CUDA / Slurm info ===")
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"[GPU {i}] {props.name}, {props.total_memory / (1024**3):.1f} GB")

        return device


# ------------------------------------------------------------
# Run helper: MLflow + Perun wrapping
# ------------------------------------------------------------
def run_with_mlflow_and_perun(trainer: BaseTrainer) -> float:
    cfg = trainer.cfg

    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        # basic params from config
        mlflow.log_params(asdict(cfg))

        # device / run id
        mlflow.log_param("device", str(trainer.device))
        mlflow.log_param("run_identifier", trainer.run_identifier)

        # model/dataset-specific params
        extra = trainer.extra_params()
        if extra:
            mlflow.log_params(extra)

        # wrap training loop with Perun
        wrapped_train = perun.perun(
            data_out=str(trainer.perun_out_dir),
            format="json",
        )(trainer.train_loop)

        try:
            perun.register_callback(log_perun_metrics_to_mlflow)
        except Exception as e:
            print(f"Perun callback registration failed: {e}")

        best_val = wrapped_train()

        # log final model
        mlflow.pytorch.log_model(trainer.get_model_for_logging(), artifact_path="final_model")

        if best_val is not None:
            mlflow.log_metric("best_val_loss", best_val)

    return best_val
