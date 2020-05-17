from __future__ import absolute_import, division, print_function, unicode_literals
import math
from typing import Iterable
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler

from latmats.utils import parse_formula, elements
from latmats.pretraining.model import Word2VecPretrainingModel
from latmats.tasks.baselines import BaseTesterEstimator


class RegressionModel(BaseTesterEstimator):

    def __init__(self, name: str, pretraining_model: Word2VecPretrainingModel, quiet: bool = False):
        self.pretraining_model = pretraining_model
        self.pretraining_model.compile()
        self.pretraining_model.load_weights()
        self.scaler = StandardScaler()
        self.quiet = quiet

        self.regression_model = None
        self.regression_model_file = f"regression_model_{name}"

    def fit(self, x: Iterable[str], y) -> None:
        self._qprint("fitting regression model")
        features = self._convert_formulas_to_matrices(x)
        targets = np.asarray(y)
        self.scaler.fit(targets.reshape((-1, 1)))
        targets = self.scaler.transform(targets.reshape((-1, 1))).reshape((-1))
        targets, features = shuffle(targets, features)

        self._compile()

        # Define the Keras TensorBoard callback.
        logdir = "logdir_regression/fit/"
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
        self.regression_model.fit(features, targets, validation_split=0.20, epochs=10000, callbacks=[tensorboard_callback, early_stopping_callback], batch_size=128)
        self._qprint("regression model fit")

    def predict(self, x: Iterable[str]) -> Iterable:
        features = self._convert_formulas_to_matrices(x)
        y_pred = self.regression_model.predict(features)
        print("y_pred", y_pred.shape, type(y_pred))

        y_pred_rescaled = self.scaler.transform(y_pred.reshape((-1, 1))).reshape((-1))
        return y_pred_rescaled

    def save_weights(self):
        self._qprint(f"saving regression model weights to {self.regression_model_file}")
        self.regression_model.save_weights(self.regression_model_file)
        self._qprint("regression model weights saved")

    def load_weights(self):
        self._qprint(f"loading regression model weights from {self.regression_model_file}")
        self.regression_model.load_weights(self.regression_model_file)
        self._qprint("loaded regression model weights")

    @staticmethod
    def _convert_formulas_to_matrices(x):
        features = []
        for xi in x:
            features.append(parse_formula(xi))
        features = np.asarray(features)
        return features

    def _compile(self):
        self._qprint("compiling regression model")
        # Input for one material is a matrix of size max_material_length x n_elements + 1
        # Each embeds ONE component element's absolute count, final column is fraction of total
        # Pad with 0s to reach max_length
        max_len = self.pretraining_model.max_len
        input_matrices = tf.keras.layers.Input(shape=(max_len, len(elements)))

        # Output is dimension 2 because we're using fancy robust losses
        # Interpret first value as prediction for quantity (ie Ef)
        # And second as an estimate of uncertainty
        # Allows the model to indicate levels of certainty
        output = tf.keras.layers.Dense(units=2)(self.pretraining_model.model_mat2vec_hiddenrep(input_matrices))
        regression_model = tf.keras.Model(inputs=input_matrices, outputs=output)
        optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-7)
        regression_model.compile(optimizer=optimizer, loss=self._robust_l1, metrics=[self._unnormalized_mae])
        self.regression_model = regression_model
        self._qprint("regression model compiled")

    def _unnormalized_mae(self, y_true, y_pred):
        # Define a function that gives us the MAE of unnormalized predictions to monitor during training
        mean, std = self.scaler.mean_, self.scaler.scale_
        unnorm_y_pred = y_pred * std + mean
        unnorm_y_true = y_true * std + mean

        mae = tf.abs(unnorm_y_pred[:, 0] - unnorm_y_true[:, 0])

        return tf.reduce_mean(mae)

    def _robust_l1(self, x_true, x_pred):
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
        loss = np.sqrt(2.0) * tf.abs(output - target) * tf.exp(-std) + std
        return tf.reduce_mean(loss)

    def _robust_l2(self, x_true, x_pred):
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

    def _qprint(self, str):
        print(str) if not self.quiet else None
