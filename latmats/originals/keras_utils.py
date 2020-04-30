import tensorflow as tf
import numpy as np
import math
import re
from collections import defaultdict
from tensorflow.keras.layers import Layer

elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Pa", "Al", "Np", "Am", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "U", "Pu", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

formulare = re.compile(r'([A-Z][a-z]*)(\d*)')


def parse_formula(formula):
    pairs = formulare.findall(formula)
    length = sum((len(p[0]) + len(p[1]) for p in pairs))
    assert length == len(formula)
    formula_dict = defaultdict(int)
    for el, sub in pairs:
        formula_dict[el] += float(sub) if sub else 1

    max_length = 9
    total = float(sum(formula_dict.values()))

    matrix = np.zeros((max_length, len(elements) + 1))
    for i, ent in enumerate(formula_dict.items()):
        el, num = ent
        el = elements.index(el)
        matrix[i, el] = num
        matrix[i, -1] = num / total

    return matrix

class BatchLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.
    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule, initial_lr):
        super(BatchLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.initial_lr = initial_lr

    def on_batch_end(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        # lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(batch) * self.initial_lr
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def is_material(word, material2index):
    if word in material2index.keys():
        return 1.0
    else:
        return 0.0


def fancy_sentence_sampler(sentence, window_size, material2index=None):
    sentence_length = len(sentence)
    if material2index is not None:
        mask = [is_material(w, material2index) for w in sentence]
        mask = mask / np.sum(mask)
        indices = np.arange(0, sentence_length)
        centre = np.random.choice(indices, p=mask)
    else:
        centre = np.random.randint(0, sentence_length)

    if centre <= window_size:
        min_value = 0
    else:
        min_value = centre - window_size
    if centre + window_size + 1 > sentence_length:
        max_value = sentence_length
    else:
        max_value = centre + window_size + 1

    indices = np.arange(min_value, max_value)
    probs = list(reversed(np.arange(1, centre - min_value + 1))) + \
        [0] + list(np.arange(1, max_value - centre))

    probs = softmax(probs)

    index = np.random.choice(indices, p=probs)

    context = sentence[index]
    target = sentence[centre]
    return target, context


def example_generator_not_material(corpus, window_size, word2index):
    n_examples = len(corpus)

    while True:
        abstract = corpus[np.random.randint(0, n_examples)]

        target, context = fancy_sentence_sampler(abstract, window_size)

        # Dummy output
        yield (word2index[target], word2index[context]), 0


def example_generator_material(corpus, window_size, word2index, material2index):
    n_examples = len(corpus)

    while True:
        while True:
            abstract = corpus[np.random.randint(0, n_examples)]
            if any([is_material(w, material2index) == 1 for w in abstract]):
                break

        target, context = fancy_sentence_sampler(
            abstract, window_size, material2index=material2index)

        material = parse_formula(target)
        # print(material, fraction, word2index[context])
        yield (material, word2index[context]), 0


def scaled_dot_product_attention(query, key, value):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs[
            'value']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value)

        self.attention_weights = attention_weights

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


class FractionalEncoding(Layer):

    def __init__(self, d_model):
        super(FractionalEncoding, self).__init__()
        self.d_model = d_model

    def get_angles(self, fraction, i):
        angles = 1 / tf.pow(tf.cast(1000, tf.float32), (2 * (i // 2)) /
                            tf.cast(self.d_model, tf.float32))
        return fraction * angles

    def positional_encoding(self, fraction):
        angle_rads = tf.keras.backend.map_fn(
            lambda x: self.get_angles(tf.pow(fraction, -1), x), tf.range(self.d_model, dtype=tf.float32))
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[1::2])
        fractional_encoding = tf.concat([sines, cosines], axis=0)
        return tf.cast(fractional_encoding, tf.float32)

    def call(self, inputs):
        material_embedding = inputs[0]
        fractions = inputs[1]
        fraction_encodings = tf.keras.backend.map_fn(
            lambda x: self.positional_encoding(x), fractions)
        return material_embedding + tf.transpose(fraction_encodings, perm=[0, 2, 1])


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def edge_matrix(max_len):
    edge_matrix = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True)
    write_idx = 0
    for i in tf.range(max_len):
        for j in tf.range(i, dtype=tf.int32):
            edge_matrix.write(write_idx, tf.stack([i, j], axis=-1))
            write_idx += 1
        for j in tf.range(i + 1, max_len, dtype=tf.int32):
            edge_matrix.write(write_idx, tf.stack([i, j], axis=-1))
            write_idx += 1

    # edge_matrix = tf.stack(edge_list, axis=0)
    return edge_matrix


class WeightedSoftAttentionMessage(Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(WeightedSoftAttentionMessage, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        self.soft_attention_layers = []
        self.attention_contribution_layers = []

        for i in range(num_heads * 2):
            self.soft_attention_layers.append(tf.keras.layers.Dense(
                units=256, activation=leaky_relu, kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
            self.soft_attention_layers.append(tf.keras.layers.Dense(
                units=1, activation='linear', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))

            self.attention_contribution_layers.append(tf.keras.layers.Dense(
                units=256, activation=leaky_relu, kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
            self.attention_contribution_layers.append(tf.keras.layers.Dense(
                units=d_model, activation='linear', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))

    def call(self, inputs):
        element, fraction, element_indices = inputs['element'], inputs['fraction'], inputs['element_indices']

        message_passed = tf.map_fn(
            lambda x: self.message_passing(x[0], x[1], x[2]), (element, fraction, element_indices), dtype=tf.float32)

        return message_passed

    @tf.function
    def message_passing(self, element, fraction, element_indices):
        # print(element)
        # concatenated = tf.concat([element, element], axis=-1)

        element_length = tf.math.count_nonzero(fraction, dtype=tf.int32)

        # node_matrix = edge_matrix(element_length).concat()

        # print(element)
        # print(node_matrix)
        message_inputs = tf.gather(element, tf.cast(element_indices, tf.int32))
        concatenated = tf.reshape(
            message_inputs, (element_length, element_length, 2 * self.d_model))

        # concatenated = tf.multiply(tf.expand_dims(
        #     element, axis=1), tf.expand_dims(element, axis=2))

        attention_contributions = []
        for i in range(self.num_heads):
            attention_weights = self.soft_attention_layers[2 * i](
                concatenated)
            attention_weights = self.soft_attention_layers[2 * i + 1](
                attention_weights)

            attention_weights = tf.exp(attention_weights)
            # print(attention_weights)
            logits = tf.multiply(attention_weights,
                                 tf.expand_dims(tf.expand_dims(fraction[:element_length], axis=0), axis=-1))

            logits = logits / tf.reduce_sum(logits, axis=-2, keepdims=True)

            attention_contribution = self.attention_contribution_layers[2 * i](
                concatenated)
            attention_contribution = self.attention_contribution_layers[2 * i + 1](
                attention_contribution)

            attention_contribution = tf.multiply(
                attention_contribution, logits)

            attention_contribution = tf.reduce_sum(
                attention_contribution, axis=1)

            attention_contributions.append(attention_contribution)

        averaged_contribution = tf.keras.layers.Average()(
            attention_contributions)

        return element + tf.pad(averaged_contribution, [[0, tf.shape(fraction)[0] - element_length], [0, 0]])


def penalized_tanh(x):
    alpha = 0.25
    return tf.maximum(tf.nn.tanh(x), alpha * tf.nn.tanh(x))


def leaky_relu(x):
    alpha = 0.01
    return tf.maximum(x, alpha * x)


class WeightedSoftAttention(Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(WeightedSoftAttention, self).__init__(name=name)
        self.d_model = d_model

        self.soft_attention_layers = []
        self.attention_contribution_layers = []
        self.soft_attention_layers.append(tf.keras.layers.Dense(
            units=256, activation=leaky_relu, kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
        self.soft_attention_layers.append(tf.keras.layers.Dense(
            units=1, activation='linear', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))

        self.attention_contribution_layers.append(tf.keras.layers.Dense(
            units=256, activation=leaky_relu, kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
        self.attention_contribution_layers.append(tf.keras.layers.Dense(
            units=d_model, activation='linear', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))

        self.batch_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)

    def call(self, inputs):
        element, fraction = inputs['element'], inputs['fraction']

        attention_weights = self.soft_attention_layers[0](
            element)
        attention_weights = self.soft_attention_layers[1](
            attention_weights)

        attention_weights = self.batch_normalization(attention_weights)
        attention_weights = tf.exp(attention_weights)
        logits = tf.multiply(attention_weights,
                             tf.expand_dims(fraction, axis=-1))

        logits = logits / tf.reduce_sum(logits, axis=-2, keepdims=True)

        attention_contribution = self.attention_contribution_layers[0](
            element)
        attention_contribution = self.attention_contribution_layers[1](
            attention_contribution)
        attention_contribution = tf.reduce_sum(tf.multiply(
            attention_contribution, logits), axis=1)

        # print(attention_contribution)
        # attention_contribution = tf.reduce_sum(
        #     attention_contribution, axis=2)

        return attention_contribution