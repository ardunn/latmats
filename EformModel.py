#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import json
import numpy as np
from pymatgen import Composition
from sklearn.model_selection import KFold
from keras_utils import parse_formula, elements
from sklearn.utils import shuffle
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler


max_len = 9
d_model = 128
n_layers = 1
n_heads = 4
dropout = 0.1

# Input for one material is a matrix of size max_material_length x n_elements + 1
# Each embeds ONE component element's absolute count, final column is fraction of total
# Pad with 0s to reach max_length
input_matrices = tf.keras.layers.Input(
    shape=(max_len, len(elements) + 1))


# Create a mask for padding rows so that we can set their contribution to 0
def create_padding_mask(input_matrices):

  mask = tf.reduce_sum(input_matrices, axis=-1, keepdims=False)
  mask = tf.cast(tf.math.equal(mask, 0), dtype=tf.float32)
  # output mask should be dimension 4 because it's used in the attention layer
  return mask[:, tf.newaxis, tf.newaxis, :]


mask = tf.keras.layers.Lambda(
    create_padding_mask, output_shape=(1, 1, max_len),
    name='enc_padding_mask')(input_matrices)

# transform to a d_model length representation with a linear layer
attention = tf.keras.layers.Dense(
    units=d_model)(input_matrices)


def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding rows
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output


def attention_layer(d_model, n_heads, name="encoder_layer"):
  inputs = tf.keras.layers.Input(shape=(max_len, d_model), name="inputs")

  input_mask = tf.keras.layers.Input(shape=(1, 1, max_len,), name="mask")
  inputs_norm = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs)

  batch_size = tf.shape(inputs)[0]

  def split_heads(inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, n_heads, d_model // n_heads))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  # Let query, key, and value transformations be set independently
  query = tf.keras.layers.Dense(units=d_model)(inputs_norm)

  key = tf.keras.layers.Dense(units=d_model)(inputs_norm)

  value = tf.keras.layers.Dense(units=d_model)(inputs_norm)

  query = split_heads(query, batch_size)
  key = split_heads(key, batch_size)
  value = split_heads(value, batch_size)

  scaled_attention = scaled_dot_product_attention(
      query, key, value, input_mask)

  scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

  # concatenation of heads
  concat_attention = tf.reshape(scaled_attention,
                                (batch_size, -1, d_model))

  attention = tf.keras.layers.Dense(units=d_model)(concat_attention)

  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs_norm + attention)

  outputs = tf.keras.layers.PReLU()(attention)

  return tf.keras.Model(
      inputs=[inputs, input_mask], outputs=outputs, name=name)


for i in range(n_layers):
  # Attention layers
  attention = attention_layer(
      d_model, n_heads, name=str(i + 1))([attention, mask])

# Output of attention layers is d_model * n_elements
# Average over elements to get a d_model representation for each material
# This is standard for text-based attention models for e.g. classification to create an embedding for the entire text
# But maybe room for improvement here?
attention = tf.keras.layers.GlobalAveragePooling1D(
    data_format='channels_last')(attention)

# A couple of fully connected layers to transform to output dimension
attention = tf.keras.layers.Dense(
    units=d_model / 2, activation=None)(attention)

attention = tf.keras.layers.PReLU()(attention)

model_hidden_rep = tf.keras.Model(
    inputs=input_matrices, outputs=attention)

# Output is dimension 2 because we're using fancy robust losses
# Interpret first value as prediction for quantity (ie Ef)
# And second as an estimate of uncertainty
# Allows the model to indicate levels of certainty
output = tf.keras.layers.Dense(
    units=2)(model_hidden_rep(input_matrices))


regression_model = tf.keras.Model(
    inputs=input_matrices, outputs=output)

# Ef for ground state materials courtesy Chris
input_file = "hullout.json"
print("Reading input data from {}".format(input_file))

with open(input_file, 'r') as f:
  input_data = json.load(f)

print("Preprocessing data")

labels = []
targets = []
features = []

for mat, v in input_data.items():
  try:
    mat_clean = mat
    # Transform each string material to its stoichiometry matrix representation
    feature = parse_formula(mat_clean)
    if not mat in labels:
      labels.append(mat_clean)
      targets.append(v['Ef'])
      features.append(feature)
  except (ValueError, TypeError):
    pass

labels = np.array(labels)
targets = np.array(targets)
features = np.array(features)
labels, targets, features = shuffle(
    labels, targets, features)


kf = KFold(n_splits=5, shuffle=True, random_state=10)


def RobustL1(x_true, x_pred):
  """
  Robust L1 loss using a lorentzian prior. Allows for estimation
  of an aleatoric uncertainty.
  """
  # Rhys wrote this comment. I had to look up 'aleatoric'. It's pretty simple at core though.
  # First output is the output value of interest
  output = x_pred[:, 0]
  # Second is the standard deviation of the prediction uncertainty
  std = x_pred[:, 1]
  target = x_true[:, 0]
  # Regular absolute error, scaled by exp(-std) allows for smaller losses from cases where the model knows it's uncertain
  # Then add on std to make sure it knows it should try to minimise std instead of just declaring that it knows nothing about anything
  loss = np.sqrt(2.0) * tf.abs(output - target) * \
      tf.exp(-std) + std

  return tf.reduce_mean(loss)


def RobustL2(x_true, x_pred):
  """
  Robust L2 loss using a gaussian prior. Allows for estimation
  of an aleatoric uncertainty.
  """
  output = x_pred[:, 0]
  log_std = x_pred[:, 1]
  target = x_true[:, 0]

  loss = 0.5 * tf.pow(output - target, 2.0) * \
      tf.exp(- 2.0 * log_std) + log_std
  return tf.reduce_mean(loss)


for train_indices, test_indices in kf.split(labels):

  # optimizer = tfa.optimizers.Yogi(learning_rate=5e-4)
  # Regular adam or adamw work fine
  optimizer = tfa.optimizers.AdamW(

      learning_rate=5e-4, weight_decay=1e-6)
  targets_train = targets[train_indices]
  labels_train = labels[train_indices]
  features_train = features[train_indices]

  targets_test = targets[test_indices]
  labels_test = labels[test_indices]
  features_test = features[test_indices]

  # Normalize target values
  normalizer = StandardScaler().fit(targets_train.reshape((-1, 1)))
  targets_train = normalizer.transform(
      targets_train.reshape((-1, 1))).reshape((-1))

  mean, std = normalizer.mean_, normalizer.scale_

  def unnormalized_mae(y_true, y_pred):
    # Define a function that gives us the MAE of unnormalized predictions to monitor during training
    unnorm_y_pred = y_pred * std + mean
    unnorm_y_true = y_true * std + mean

    mae = tf.abs(unnorm_y_pred[:, 0] - unnorm_y_true[:, 0])

    return tf.reduce_mean(mae)

  optimizer = tfa.optimizers.AdamW(
      learning_rate=5e-3, weight_decay=1e-6)

  regression_model.compile(optimizer=optimizer,
                           loss=RobustL1, metrics=[unnormalized_mae])

  model_hidden_rep.summary()
  regression_model.summary()

  def cyclical_lr(period=10, cycle_mul=0.1, tune_mul=0.05, initial_lr=5e-3):
      # Scaler: we can adapt this if we do not want the triangular CLR
    def scaler(x): return 1.

    # Lambda function to calculate the LR
    def lr_lambda(it):
      lr = initial_lr * (cycle_mul +
                         (1. - cycle_mul) * relative(it, period))
      return lr * tf.math.exp(-0.001 * it)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
      cycle = math.floor(1 + it / (period))
      x = abs(2 * (it / period - cycle) + 1)
      return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

  callback_lr = tf.keras.callbacks.LearningRateScheduler(cyclical_lr())

  # model_hidden_rep.load_weights("hidden_rep0.keras")

  regression_model.fit(features_train, targets_train,
                       validation_split=0.2, epochs=5000, callbacks=[callback_lr], batch_size=128)

  print("\n\n\n")
  print(regression_model.evaluate(features_test, targets_test))
  print("\n\n\n")
  break
