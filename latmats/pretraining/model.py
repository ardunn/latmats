#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import Counter
import json
import os

from latmats.pretraining.data_loader import load_file
from latmats.utils import example_generator_not_material, example_generator_material, elements

#todo: subSample not implemented

word2index = load_file('word2index_3mil.pkl')
index2word = load_file('index2word_3mil.pkl')
material2index = load_file('material2index.pkl')
index2material = load_file('index2material.pkl')
abstracts_3mil = load_file("abstracts_3mil.txt", as_lines=True)
processed_abstracts = load_file("processed_abstracts.txt", as_lines=True)

vocab_size = len(word2index.keys())
embedding_dimension = 200
max_len = 9
d_model = 128
n_layers = 1
n_heads = 4
dropout = 0.1
window_size = 5
neg_samples = 10
dropout = 0.1
n_elements = len(elements) + 1


def negative_samples_contract(x):
    return tf.einsum('ijk,lk->ijl', x[0], x[1])


def positive_loss(ypred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(ypred), logits=ypred)


def negative_loss(ypred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(ypred), logits=ypred)


def create_padding_mask(input_matrices):
    mask = tf.reduce_sum(input_matrices, axis=-1, keepdims=False)
    mask = tf.cast(tf.math.equal(mask, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # attention_weights = tf.keras.layers.Dropout(rate=dropout)(attention_weights)
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

    query = tf.keras.layers.Dense(units=d_model)(inputs_norm)
    # query = tf.keras.layers.PReLU()(query)
    # query = tf.keras.layers.Dense(units=d_model)(query)

    key = tf.keras.layers.Dense(units=d_model)(inputs_norm)
    # key = tf.keras.layers.PReLU()(key)
    # key = tf.keras.layers.Dense(units=d_model)(key)

    value = tf.keras.layers.Dense(units=d_model)(inputs_norm)
    # value = tf.keras.layers.PReLU()(value)
    # value = tf.keras.layers.Dense(units=d_model)(value)

    query = split_heads(query, batch_size)
    key = split_heads(key, batch_size)
    value = split_heads(value, batch_size)

    scaled_attention = scaled_dot_product_attention(query, key, value, input_mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, d_model))

    attention = tf.keras.layers.Dense(units=d_model)(concat_attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs_norm + attention)
    outputs = tf.keras.layers.PReLU()(attention)

    return tf.keras.Model(inputs=[inputs, input_mask], outputs=outputs, name=name)


class Word2VecPretrainingModel:

    def __init__(self, quiet=False):
        self.quiet = quiet
        self.model_mat2vec = None
        self.model_word2vec = None
        self.model_mat2vec_hiddenrep = None

        thisdir = os.path.dirname(os.path.abspath(__file__))
        self.model_mat2vec_weights_file = os.path.join(thisdir, "mat2vec.keras")
        self.model_word2vec_weights_file = os.path.join(thisdir, "word2vec.keras")
        self.model_mat2vec_hiddenrep_weights_file = os.path.join(thisdir, "mat2vec_hiddenrep.keras")

    def compile(self):
        if not self.quiet:
            print("compiling model")
        input_context = tf.keras.Input(shape=(1,), name="inputs_context")
        embeddings_context = tf.keras.layers.Embedding(vocab_size, embedding_dimension, name="embeddings_context")
        sample_context_embedding = embeddings_context(input_context)

        input_target = tf.keras.Input(shape=(1,), name="inputs_target")
        embeddings_target = tf.keras.layers.Embedding(vocab_size, embedding_dimension, name="embeddings_target")
        sample_target_embedding = embeddings_target(input_target)

        negative_sample_distribution = tfp.distributions.Categorical(logits=np.arange(vocab_size))
        negative_samples = negative_sample_distribution.sample(sample_shape=(neg_samples,))
        negative_target_embeddings = embeddings_target(negative_samples)

        positive_probability = tf.keras.layers.Dot(axes=(2, 2))([sample_context_embedding, sample_target_embedding])
        negative_probability = tf.keras.layers.Lambda(negative_samples_contract)([sample_context_embedding, negative_target_embeddings])

        positive_loss_tensor = tf.keras.layers.Lambda(positive_loss)(positive_probability)
        negative_loss_tensor = tf.keras.layers.Lambda(negative_loss)(negative_probability)

        loss = tf.keras.layers.Concatenate()([positive_loss_tensor, negative_loss_tensor])
        loss = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.sum(x, axis=2, keepdims=False), axis=1))(loss)

        model_word2vec = tf.keras.Model(inputs=[input_target, input_context], outputs=loss)

        input_matrices = tf.keras.layers.Input(shape=(max_len, len(elements) + 1))
        mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, max_len), name='enc_padding_mask')(input_matrices)
        attention = tf.keras.layers.Dense(units=d_model)(input_matrices)

        for i in range(n_layers):
            attention = attention_layer(d_model, n_heads, name=str(i + 1))([attention, mask])
        attention = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(attention)

        attention = tf.keras.layers.Dense(units=d_model / 2, activation=None)(attention)
        attention = tf.keras.layers.PReLU()(attention)

        model_mat2vec_hiddenrep = tf.keras.Model(inputs=input_matrices, outputs=attention)
        hidden_rep = model_mat2vec_hiddenrep(input_matrices)
        sample_material_embedding = tf.keras.layers.Dense(units=embedding_dimension, activation=None)(hidden_rep)
        sample_material_embedding = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(sample_material_embedding)

        positive_probability_material = tf.keras.layers.Dot(axes=(2, 2))([sample_context_embedding, sample_material_embedding])
        positive_loss_material_tensor = tf.keras.layers.Lambda(positive_loss)(positive_probability_material)

        loss_material = tf.keras.layers.Concatenate()([positive_loss_material_tensor, negative_loss_tensor])
        loss_material = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.sum(x, axis=2, keepdims=False), axis=1))(loss_material)

        model_word2vec.compile(optimizer='adam', loss='mse')

        model_mat2vec = tf.keras.Model(inputs=[input_matrices, input_context], outputs=loss_material)
        model_mat2vec.compile(optimizer='adam', loss='mse')

        self.model_word2vec = model_word2vec                     # outputs word2vec embedding
        self.model_mat2vec = model_mat2vec                       # outputs word embedding of material
        self.model_mat2vec_hiddenrep = model_mat2vec_hiddenrep   # outputs attention layer of model_mat2vec

        if not self.quiet:
            print("model compiled.")

    def summarize(self):
        self.model_word2vec.summary()
        self.model_mat2vec.summary()
        self.model_mat2vec_hiddenrep.summary()

    def train(self):
        # word_count = Counter()
        # for i, l in enumerate(abstracts_3mil):
        #     if i % 100000 == 0:
        #         print(i)
        #     abstract = l.strip().split()
        #     word_count.update(abstract)

        if not self.quiet:
            print("generating pretraining datasets")

        corpus = []
        for abstract in processed_abstracts:
            corpus.append(json.loads(abstract))
        dataset = tf.data.Dataset.from_generator(lambda: example_generator_not_material(corpus, window_size, word2index), ((tf.int64, tf.int64), tf.int64), output_shapes=((tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([]))).prefetch(tf.data.experimental.AUTOTUNE).batch(512)
        dataset_material = tf.data.Dataset.from_generator(lambda: example_generator_material(corpus, window_size, word2index, material2index), ((tf.float32, tf.int64), tf.int64), output_shapes=((tf.TensorShape([9, n_elements]), tf.TensorShape([])), tf.TensorShape([]))).batch(512).prefetch(tf.data.experimental.AUTOTUNE)


        if not self.quiet:
            print("completed pretraining datasets")
            print("training model...")

        n_training_cyles = 20
        for i in range():
            if not self.quiet:
                print(f"training cycle {i}/{n_training_cyles}")
            self.model_word2vec.fit(dataset, steps_per_epoch=1000, epochs=1)
            self.model_mat2vec.fit(dataset_material, steps_per_epoch=1000, epochs=1)
            # self.model_mat2vec_hiddenrep.save_weights("mat2vec_hiddenrep{}.keras".format(i))

    def save_weights(self):
        self.model_word2vec.save_weights(self.model_word2vec_weights_file)
        self.model_mat2vec.save_weights(self.model_mat2vec_weights_file)
        self.model_mat2vec_hiddenrep.save_weights(self.model_mat2vec_hiddenrep_weights_file)

    def load_weights(self):
        if not self.quiet:
            print("loading model weights")
        self.model_word2vec.load_weights(self.model_word2vec_weights_file)
        self.model_mat2vec.load_weights(self.model_mat2vec_weights_file)

        if not self.quiet:
            print("model weights loaded")
        # print(model_material([[1], [1.]))


if __name__ == "__main__":
    pass















