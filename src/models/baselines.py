"""
Baseline lightweight models — Section 3.6.

MobileNetV2 and SqueezeNet, adapted to 64x64 / 43 classes.
These are trained under *identical* conditions to the proposed model,
including degradation-aware training (Fix #3) so the comparison is fair.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2 as KerasMobileNetV2


def build_mobilenetv2(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 43,
    alpha: float = 1.0,
    dropout: float = 0.30,
    weights: str | None = None,
) -> keras.Model:
    """
    MobileNetV2 backbone (Sandler et al., 2018) with custom head.
    alpha scales width; weights=None trains from scratch (no ImageNet).
    """
    base = KerasMobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        pooling=None,
        weights=weights,
    )
    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = keras.Model(base.input, outputs, name="mobilenetv2_baseline")
    return model


# ---- SqueezeNet (Iandola et al., 2016) — compact re-implementation

def _fire_module(x, squeeze_filters: int, expand_1x1: int, expand_3x3: int, name: str):
    s = layers.Conv2D(squeeze_filters, 1, padding="same", activation="relu", name=f"{name}_squeeze")(x)
    e1 = layers.Conv2D(expand_1x1, 1, padding="same", activation="relu", name=f"{name}_expand1x1")(s)
    e3 = layers.Conv2D(expand_3x3, 3, padding="same", activation="relu", name=f"{name}_expand3x3")(s)
    return layers.Concatenate(name=f"{name}_concat")([e1, e3])


def build_squeezenet(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 43,
    dropout: float = 0.30,
) -> keras.Model:
    """
    SqueezeNet v1.0-style network with fire modules and late downsampling.
    Adapted for 64x64 input.
    """
    inputs = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    x = _fire_module(x, 16, 64, 64, "fire2")
    x = _fire_module(x, 16, 64, 64, "fire3")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool3")(x)

    x = _fire_module(x, 32, 128, 128, "fire4")
    x = _fire_module(x, 32, 128, 128, "fire5")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool5")(x)

    x = _fire_module(x, 48, 192, 192, "fire6")
    x = _fire_module(x, 48, 192, 192, "fire7")
    x = _fire_module(x, 64, 256, 256, "fire8")

    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Conv2D(num_classes, 1, padding="same", activation="relu", name="conv10")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Activation("softmax", name="pred")(x)

    model = keras.Model(inputs, outputs, name="squeezenet_baseline")
    return model
