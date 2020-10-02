import os
import json

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from latmats.pretraining.data_loader import load_file
from latmats.utils import example_generator_not_material, example_generator_material, elements


class Word2VecPretrainingModel:
    def __init__(self,
                 quiet=False,
                 name=None,
                 embedding_dimension=200,
                 d_model=128,
                 max_len=9,
                 n_layers=1,
                 n_heads=4,
                 dropout=0.1,
                 window_size=5,
                 neg_samples=10,
                 use_attention=True,
                 ):
        self.quiet = quiet
        self.model_mat2vec = None
        self.model_word2vec = None
        self.model_mat2vec_hiddenrep = None
        self.name = "no name" if not name else name

        suffix = f"_{name}" if name else ""

        thisdir = os.path.dirname(os.path.abspath(__file__))
        self.model_mat2vec_weights_file = os.path.join(thisdir, f"mat2vec{suffix}.keras")
        self.model_word2vec_weights_file = os.path.join(thisdir, f"word2vec{suffix}.keras")
        self.model_mat2vec_hiddenrep_weights_file = os.path.join(thisdir, f"mat2vec_hiddenrep{suffix}.keras")


        self.embedding_dimension = embedding_dimension
        self.d_model = d_model
        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.use_attention = use_attention

        self.word2index = None

    def compile(self):
        self._qprint("compiling model")

        self.word2index = load_file('word2index_3mil.pkl', quiet=self.quiet)
        vocab_size = len(self.word2index.keys())

        input_context = tf.keras.Input(shape=(1,), name="inputs_context")
        embeddings_context = tf.keras.layers.Embedding(vocab_size, self.embedding_dimension, name="embeddings_context")
        sample_context_embedding = embeddings_context(input_context)

        input_target = tf.keras.Input(shape=(1,), name="inputs_target")
        embeddings_target = tf.keras.layers.Embedding(vocab_size, self.embedding_dimension, name="embeddings_target")
        sample_target_embedding = embeddings_target(input_target)

        negative_sample_distribution = tfp.distributions.Categorical(logits=np.arange(vocab_size))
        negative_samples = negative_sample_distribution.sample(sample_shape=(self.neg_samples,))
        negative_target_embeddings = embeddings_target(negative_samples)

        positive_probability = tf.keras.layers.Dot(axes=(2, 2))([sample_context_embedding, sample_target_embedding])
        negative_probability = tf.keras.layers.Lambda(self._negative_samples_contract)([sample_context_embedding, negative_target_embeddings])

        positive_loss_tensor = tf.keras.layers.Lambda(self._positive_loss)(positive_probability)
        negative_loss_tensor = tf.keras.layers.Lambda(self._negative_loss)(negative_probability)

        loss = tf.keras.layers.Concatenate()([positive_loss_tensor, negative_loss_tensor])
        loss = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.sum(x, axis=2, keepdims=False), axis=1))(loss)

        model_word2vec = tf.keras.Model(inputs=[input_target, input_context], outputs=loss)

        input_matrices = tf.keras.layers.Input(shape=(self.max_len, len(elements)))
        attention = tf.keras.layers.Dense(units=self.d_model)(input_matrices)

        mask = tf.keras.layers.Lambda(self._create_padding_mask, output_shape=(1, 1, self.max_len), name='enc_padding_mask')(input_matrices)

        for i in range(self.n_layers):
            if self.use_attention:
                attention = self._attention_layer(self.d_model, self.n_heads, name=str(i + 1))([attention, mask])
            else:
                # attention = tf.keras.layers.Dense(units=self.d_model, activation="relu")(attention)
                attention = tf.keras.layers.LeakyReLU(alpha=0.05)(attention)

        # Output of attention layers is d_model * n_elements
        # Average over elements to get a d_model representation for each material
        # This is standard for text-based attention models for e.g. classification to create an embedding for the entire text
        # But maybe room for improvement here?
        attention = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(attention)

        attention = tf.keras.layers.Dense(units=self.d_model / 2, activation=None)(attention)
        attention = tf.keras.layers.PReLU()(attention)

        model_mat2vec_hiddenrep = tf.keras.Model(inputs=input_matrices, outputs=attention)
        hidden_rep = model_mat2vec_hiddenrep(input_matrices)
        sample_material_embedding = tf.keras.layers.Dense(units=self.embedding_dimension, activation=None)(hidden_rep)
        sample_material_embedding = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(sample_material_embedding)

        positive_probability_material = tf.keras.layers.Dot(axes=(2, 2))([sample_context_embedding, sample_material_embedding])
        positive_loss_material_tensor = tf.keras.layers.Lambda(self._positive_loss)(positive_probability_material)

        loss_material = tf.keras.layers.Concatenate()([positive_loss_material_tensor, negative_loss_tensor])
        loss_material = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.sum(x, axis=2, keepdims=False), axis=1))(loss_material)

        model_word2vec.compile(optimizer='adam', loss='mse')


        # todo: rename models
        model_mat2vec = tf.keras.Model(inputs=[input_matrices, input_context], outputs=loss_material)

        model_mat2vec.compile(optimizer='adam', loss='mse')

        self.model_word2vec = model_word2vec                     # outputs word2vec embedding
        self.model_mat2vec = model_mat2vec                       # outputs word embedding of material
        self.model_mat2vec_hiddenrep = model_mat2vec_hiddenrep   # outputs attention layer of model_mat2vec

        self._qprint("model compiled.")

    def summarize(self):
        self.model_word2vec.summary()
        self.model_mat2vec.summary()
        self.model_mat2vec_hiddenrep.summary()

    def train(self, only_mat2vec=False):
        self._qprint("generating pretraining datasets...")

        # corpus = []
        corpus = load_file("processed_abstracts.txt", quiet=self.quiet)
        # for abstract in processed_abstracts:
        #     corpus.append(json.loads(abstract))

        material2index = load_file('material2index.pkl', quiet=self.quiet)
        n_elements = len(elements)
        dataset = tf.data.Dataset.from_generator(lambda: example_generator_not_material(corpus, self.window_size, self.word2index), ((tf.int64, tf.int64), tf.int64), output_shapes=((tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([]))).prefetch(tf.data.experimental.AUTOTUNE).batch(512)
        dataset_material = tf.data.Dataset.from_generator(lambda: example_generator_material(corpus, self.window_size, self.word2index, material2index), ((tf.float32, tf.int64), tf.int64), output_shapes=((tf.TensorShape([9, n_elements]), tf.TensorShape([])), tf.TensorShape([]))).batch(512).prefetch(tf.data.experimental.AUTOTUNE)

        self._qprint("completed pretraining datasets\ntraining model...")


        # cyclical training can work better
        # n_training_cycles = 30
        # for i in range(n_training_cycles):
        #     self._qprint(f"training cycle {i}/{n_training_cycles}")
        #     self.model_word2vec.fit(dataset, steps_per_epoch=1000, epochs=1)
        #     self.model_mat2vec.fit(dataset_material, steps_per_epoch=1000, epochs=1)

        logdir = "../logdir_pretraining/fit/"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', min_delta=0, patience=1,
        )

        if only_mat2vec:
            self._qprint(f"reloading previously trained word2vec weights at {self.model_word2vec_weights_file}...")
            self.model_word2vec.load_weights(self.model_word2vec_weights_file)
            self._qprint("loaded previously trained word2vec weights.")
        else:
            self._qprint("training word2vec...")
            self.model_word2vec.fit(dataset, steps_per_epoch=1000, epochs=25, callbacks=[early_stopping, tensorboard_callback])
            self._qprint("word2vec trained.")

        for layer in self.model_word2vec.layers:
            layer.trainable = False

        self._qprint("training mat2vec...")
        self.model_mat2vec.fit(dataset_material, steps_per_epoch=500, epochs=20, callbacks=[early_stopping, tensorboard_callback])
        self._qprint("mat2vec trained")

    def save_weights(self):
        self.model_word2vec.save_weights(self.model_word2vec_weights_file)
        self.model_mat2vec.save_weights(self.model_mat2vec_weights_file)
        self.model_mat2vec_hiddenrep.save_weights(self.model_mat2vec_hiddenrep_weights_file)

    def load_weights(self):
        self._qprint("loading model weights")
        self.model_word2vec.load_weights(self.model_word2vec_weights_file)
        self.model_mat2vec.load_weights(self.model_mat2vec_weights_file)
        self._qprint("model weights loaded")

    @staticmethod
    def _negative_samples_contract(x):
        return tf.einsum('ijk,lk->ijl', x[0], x[1])

    @staticmethod
    def _positive_loss(ypred):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(ypred), logits=ypred)

    @staticmethod
    def _negative_loss(ypred):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(ypred), logits=ypred)

    @staticmethod
    def _create_padding_mask(input_matrices):
        mask = tf.reduce_sum(input_matrices, axis=-1, keepdims=False)
        mask = tf.cast(tf.math.equal(mask, 0), dtype=tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    @staticmethod
    def _scaled_dot_product_attention(query, key, value, mask):
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

        # attention_weights = tf.keras.layers.(rate=dropout)(attention_weights)
        output = tf.matmul(attention_weights, value)
        return output

    def _attention_layer(self, d_model, n_heads, name="encoder_layer"):
        inputs = tf.keras.layers.Input(shape=(self.max_len, d_model), name="inputs")

        input_mask = tf.keras.layers.Input(shape=(1, 1, self.max_len,), name="mask")
        inputs_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
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

        scaled_attention = self._scaled_dot_product_attention(query, key, value, input_mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, d_model))

        attention = tf.keras.layers.Dense(units=d_model)(concat_attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            inputs_norm + attention)
        outputs = tf.keras.layers.PReLU()(attention)

        return tf.keras.Model(inputs=[inputs, input_mask], outputs=outputs, name=name)

    def _qprint(self, str):
        print(str) if not self.quiet else None















