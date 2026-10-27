"""
Proposed lightweight CNN — Section 3.6.

Initial conv -> 4 depthwise-separable blocks (32,64,128,256) with
BN + ReLU6 -> GAP -> Dropout(0.3) -> Dense(43, softmax).

Depthwise-separable = DepthwiseConv2D + Pointwise Conv2D, as in
MobileNet (Howard et al., 2017). Parameter-light and fast.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _ds_block(x, filters: int, stride: int = 1, block_id: int = 0):
    # Depthwise
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=stride, padding="same", use_bias=False,
        name=f"ds_dw_{block_id}",
    )(x)
    x = layers.BatchNormalization(name=f"ds_dw_bn_{block_id}")(x)
    x = layers.ReLU(max_value=6.0, name=f"ds_dw_relu6_{block_id}")(x)
    # Pointwise
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding="same", use_bias=False, name=f"ds_pw_{block_id}")(x)
    x = layers.BatchNormalization(name=f"ds_pw_bn_{block_id}")(x)
    x = layers.ReLU(max_value=6.0, name=f"ds_pw_relu6_{block_id}")(x)
    return x


def build_proposed(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 43,
    filters: list[int] | None = None,
    dropout: float = 0.30,
) -> keras.Model:
    if filters is None:
        filters = [32, 64, 128, 256]

    inputs = keras.Input(shape=input_shape, name="input")

    # Initial standard conv
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="conv_init")(inputs)
    x = layers.BatchNormalization(name="conv_init_bn")(x)
    x = layers.ReLU(max_value=6.0, name="conv_init_relu6")(x)

    # Four DS blocks — first block strides 1, rest stride 2 for downsampling
    strides = [1, 2, 2, 2]
    for i, (f, s) in enumerate(zip(filters, strides)):
        x = _ds_block(x, f, stride=s, block_id=i)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs, outputs, name="proposed_lightweight_cnn")
    return model
