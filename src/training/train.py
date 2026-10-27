"""
Training loop — Section 3.7.

Fix #3 (ablation): degradation_aware flag controls whether the proposed
model sees degraded images during training. Baselines are *also* trained
with degradation_aware=True for a fair comparison, plus an ablation run
with degradation_aware=False for the proposed model to isolate the
effect of the strategy.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.utils.seed import set_seed


def compile_model(model: keras.Model, lr: float = 0.001) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks(
    ckpt_path: str | Path | None = None,
    lr_decay_factor: float = 0.5,
    lr_patience: int = 3,
    early_patience: int = 7,
    log_dir: str | Path | None = None,
) -> list[keras.callbacks.Callback]:
    cbs: list[keras.callbacks.Callback] = []
    cbs.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lr_decay_factor, patience=lr_patience,
            verbose=1, min_lr=1e-6,
        )
    )
    cbs.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_patience,
            restore_best_weights=True, verbose=1,
        )
    )
    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        cbs.append(
            keras.callbacks.ModelCheckpoint(
                str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1,
            )
        )
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        cbs.append(keras.callbacks.CSVLogger(str(Path(log_dir) / "training_log.csv")))
    return cbs


def train(
    model: keras.Model,
    train_ds,
    val_ds,
    epochs: int = 50,
    callbacks: list | None = None,
    verbose: int = 1,
) -> keras.callbacks.History:
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )


def count_params_mb(model: keras.Model) -> tuple[int, float]:
    n = model.count_params()
    mb = n * 4 / (1024 * 1024)  # float32
    return n, mb


def measure_latency(
    model: keras.Model,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    runs: int = 200,
    warmup: int = 20,
    batch_size: int = 1,
) -> dict[str, float]:
    dummy = np.random.rand(batch_size, *input_shape).astype(np.float32)
    for _ in range(warmup):
        _ = model.predict(dummy, verbose=0)
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }
