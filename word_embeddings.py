#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_utils import example_generator_not_material, example_generator_material
import pickle
import tensorflow_probability as tfp
from keras_utils import example_generator_not_material, example_generator_material, elements
import os
import numpy as np

word2index = pickle.load(open(
    os.path.join('word2index.pkl'), 'rb'))
index2word = pickle.load(open(
    os.path.join('index2word.pkl'), 'rb'))
material2index = pickle.load(open(
    os.path.join('material2index.pkl'), 'rb'))
index2material = pickle.load(open(
    os.path.join('index2material.pkl'), 'rb'))

vocab_size = len(word2index.keys())

d_model = 200
hidden_dim = 20
window_size = 5
neg_samples = 10
dropout = 0.1
n_elements = len(elements)

input_context = tf.keras.Input(shape=(1,), name="inputs_context")

embeddings_context = tf.keras.layers.Embedding(
    vocab_size, d_model, name="embeddings_context")

sample_context_embedding = embeddings_context(input_context)

input_target = tf.keras.Input(shape=(1,), name="inputs_target")

embeddings_target = tf.keras.layers.Embedding(
    vocab_size, d_model, name="embeddings_target")

sample_target_embedding = embeddings_target(input_target)

negative_sample_distribution = tfp.distributions.Categorical(
    logits=np.arange(vocab_size))

negative_samples = negative_sample_distribution.sample(
    sample_shape=(neg_samples,))

negative_target_embeddings = embeddings_target(negative_samples)

positive_probability = tf.keras.layers.Dot(axes=(2, 2))(
    [sample_context_embedding, sample_target_embedding])

# There is surely a nicer way to do this. All I want to do is take the dot product and contract over indices.


def negative_sammples_contract(x):
  return tf.einsum('ijk,lk->ijl', x[0], x[1])


negative_probability = tf.keras.layers.Lambda(negative_sammples_contract)(
    [sample_context_embedding, negative_target_embeddings])


def positive_loss(ypred):
  return tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(ypred), logits=ypred)


def negative_loss(ypred):
  return tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(ypred), logits=ypred)


positive_loss_tensor = tf.keras.layers.Lambda(
    positive_loss)(positive_probability)

negative_loss_tensor = tf.keras.layers.Lambda(
    negative_loss)(negative_probability)


loss = tf.keras.layers.Concatenate()(
    [positive_loss_tensor, negative_loss_tensor])

loss = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.sum(
    x,
    axis=2,
    keepdims=False
), axis=1))(loss)


model = tf.keras.Model(
    inputs=[input_target, input_context], outputs=loss)

input_material = tf.keras.layers.Input(
    shape=(n_elements,), name="inputs_material")

attention = tf.keras.layers.Attention()([input_material, input_material])
attention = tf.keras.layers.Attention()([attention, attention])
attention = tf.keras.layers.Dropout(rate=dropout)(attention)

attention = tf.keras.layers.LayerNormalization(
    epsilon=1e-6)(attention)

hidden_rep = tf.keras.layers.Dense(
    units=hidden_dim, activation='tanh')(attention)

sample_material_embedding = tf.keras.layers.Dense(
    units=d_model, activation='tanh')(attention)

positive_probability_material = tf.keras.layers.Dot(axes=(2, 1))(
    [sample_context_embedding, sample_material_embedding])

positive_loss_material_tensor = tf.keras.layers.Lambda(
    positive_loss)(positive_probability_material)

positive_loss_material_tensor = tf.keras.layers.Lambda(
    lambda x: tf.expand_dims(x, axis=1))(positive_loss_material_tensor)

loss_material = tf.keras.layers.Concatenate()(
    [positive_loss_material_tensor, negative_loss_tensor])

loss_material = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.sum(
    x,
    axis=2,
    keepdims=False
), axis=1))(loss_material)


corpus = []


def is_material(word, material2index):
  try:
    material2index[word]
    return True
  except KeyError:
    return False


with open('rel_abstracts.bpe', 'r') as f:
  for l in f:
    abstract = l.strip().split()
    corpus.append(abstract)

dataset = tf.data.Dataset.from_generator(
    lambda: example_generator_not_material(corpus, window_size, word2index), ((tf.int64, tf.int64), tf.int64), output_shapes=((tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([]))).prefetch(tf.data.experimental.AUTOTUNE).batch(
        512)

model.compile(optimizer='adam',
              loss='mse')

model.summary()


model_material = tf.keras.Model(
    inputs=[input_material, input_context], outputs=loss_material)

dataset_material = tf.data.Dataset.from_generator(
    lambda: example_generator_material(corpus, window_size, word2index, material2index), ((tf.int64, tf.int64), tf.int64), output_shapes=((tf.TensorShape([n_elements]), tf.TensorShape([])), tf.TensorShape([]))).prefetch(tf.data.experimental.AUTOTUNE).batch(
        512)

model_material.compile(optimizer='adam',
                       loss='mse')

model_material.summary()

while True:
  model_material.fit(dataset_material, steps_per_epoch=1000, epochs=1)

  model.fit(dataset, steps_per_epoch=1000, epochs=1)
