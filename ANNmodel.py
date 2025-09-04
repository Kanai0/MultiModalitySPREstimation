# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras import layers, models, Input
from keras import backend as K

kph = 9.094e-6
kcoh = 1.064e-3
kincoh = 0.5988
rho_e_water = 3.343e23  # e-/cm3
z3_62_water = 7.522
z1_86_water = 7.115

# Residual Block:
# Residual Block A: kernel size=3 (default)
# Residual Block B: kernel size=2 (specified in input arguments)
# strides=2 will not function properly, so used strides=1
def residual_block(x, filters, kernel_size=3, strides=1, name_prefix="A"):

    """
    Projection shortcut:
      - 自動判定: (strides != 1) or (in_channels != filters) のとき 1x1 Conv を適用
    """
    in_ch = K.int_shape(x)[-1]
    use_proj = (strides != 1) or (in_ch != filters)
    
    # shortcut path
    shortcut = x
    if use_proj:
        shortcut = layers.Conv1D(filters, 1, strides=strides, padding="same",
                                 name=f"{name_prefix}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # main path
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.ReLU()(x)
    return x

# B-type branch without convA/B (B1-B2-B3-B2-B1)
def build_branch_B(input_tensor, name_prefix="B"):
    x = residual_block(input_tensor, filters=64, kernel_size=2, strides=1, name_prefix=f"{name_prefix}1") # B1
    x = residual_block(x, filters=128, kernel_size=2, strides=1, name_prefix=f"{name_prefix}2") # B2
    x = residual_block(x, filters=256, kernel_size=2, strides=1, name_prefix=f"{name_prefix}3") # B3
    x = residual_block(x, filters=128, kernel_size=2, strides=1, name_prefix=f"{name_prefix}2_back") # B2
    x = residual_block(x, filters=64, kernel_size=2, strides=1, name_prefix=f"{name_prefix}1_back") # B1
    return x

# A-type branch (A1-A2-A3-A2-A1, no final conv)
def build_branch_A(input_tensor, name_prefix="A"):
    x = residual_block(input_tensor, filters=64, kernel_size=3, strides=1, name_prefix=f"{name_prefix}1")
    x = residual_block(x, filters=128, kernel_size=3, strides=1, name_prefix=f"{name_prefix}2")
    x = residual_block(x, filters=256, kernel_size=3, strides=1, name_prefix=f"{name_prefix}3")
    x = residual_block(x, filters=128, kernel_size=3, strides=1, name_prefix=f"{name_prefix}2_back")
    x = residual_block(x, filters=64, kernel_size=3, strides=1, name_prefix=f"{name_prefix}1_back")
    x = layers.GlobalAveragePooling1D()(x)
    return x

# Full model with shared ConvA + 4 branches (3B + 1A)
def build_full_model(input_shape=(40, 1)):
    inputs = Input(shape=input_shape, name="model_input")

    # Shared initial ConvA + ReLU
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', name="ConvA_shared")(inputs)
    x = layers.ReLU(name="relu_shared")(x)

    # 3 B-type and 1 A-type branches
    branch1 = build_branch_B(x, name_prefix="B1")
    branch1 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name=f"convB_final1")(branch1)
    branch1 = layers.GlobalAveragePooling1D()(branch1)

    branch2 = build_branch_B(x, name_prefix="B2")
    branch2 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name=f"convB_final2")(branch2)
    branch2 = layers.GlobalAveragePooling1D()(branch2)

    branch3 = build_branch_B(x, name_prefix="B3")
    branch3 = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', name=f"convB_final3")(branch3)
    branch3 = layers.GlobalAveragePooling1D()(branch3)
    
    branch4 = build_branch_A(x, name_prefix="A1")

    # Concatenate and FC layers
    merged = layers.Concatenate(name="concat")([branch1, branch2, branch3, branch4])
    x = layers.Dense(64, activation='relu', name="fc1")(merged)
    x = layers.Dense(64, activation='relu', name="fc2")(x)
    output = layers.Dense(1, activation='linear', name="rho_m")(x)

    model = models.Model(inputs=inputs, outputs=output, name="Chang_ResNet_Model_Corrected")
    return model


# --- Physics-based CT値予測 ---
def physics_ct_loss(y_true_ct, y_pred_rho, z_3_62_water=7.522, z_1_86_water=7.115):
    # y_pred_rho: [batch, 1] 質量密度
    num = y_pred_rho * (kph * z_3_62 + kcoh * z_1_86 + kincoh)
    den = kph * z3_62_water + kcoh * z1_86_water + kincoh
    hu_pred = 1000.0 * (num / den - 1.0)
    return tf.reduce_mean(tf.square(hu_pred - y_true_ct))

# --- カスタム損失関数 ---
class HybridLoss(tf.keras.losses.Loss):
    def __init__(self, delta, z_3_62, z_1_86):
        super().__init__()
        self.delta = delta
        self.z_3_62 = tf.constant(z_3_62, dtype=tf.float32)
        self.z_1_86 = tf.constant(z_1_86, dtype=tf.float32)

    def call(self, y_true, y_pred):
        rho_true, ct_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
        mse_rho = tf.reduce_mean(tf.square(y_pred - rho_true))
        mse_ct = physics_ct_loss(ct_true, y_pred, self.z_3_62, self.z_1_86)
        return (1 - self.delta) * mse_rho + self.delta * mse_ct
