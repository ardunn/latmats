from abc import abstractmethod
from typing import Iterable

from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty, ElementFraction
from matminer.featurizers.base import MultipleFeaturizer


class BaseTesterEstimator:
    """
    A base class for creating estimators to run in the tester (regression).

    Only requirements is that the derived class implements the fit and predict
    functions below as described in their docstrings.
    """

    @abstractmethod
    def fit(self, x: Iterable[str], y) -> None:
        """
        Fit the model to the argument data.

        Args:
            x (Iterable(str)): The compositions, as strings in an iterable.
            y ([int, float]): The target values for regression

        Returns:
            None.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Iterable[str]) -> Iterable:
        """
        Predict property for a set of inputs.

        Args:
            x (Iterable(str)): The compositions, as strings in an iterable.

        Returns:
            ([int, str]): The predicted target values, as an iterable of floats/strs.
        """
        raise NotImplementedError


class DummyEstimator(BaseTesterEstimator):

    def __init__(self):
        self.regressor = DummyRegressor()

    def fit(self, x, y):
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)


class RFEstimator(BaseTesterEstimator):

    def __init__(self, pbar=False):
        self.regressor = RandomForestRegressor(n_estimators=1000)
        self.stc = StrToComposition()
        ep = ElementProperty.from_preset("magpie")
        ef = ElementFraction()
        self.featurizer = MultipleFeaturizer([ep, ef])
        self.pbar = pbar

    def _generate_features(self, x):
        comps = [o[0] for o in self.stc.featurize_many(x, pbar=self.pbar)]
        features = np.asarray(self.featurizer.featurize_many(comps, pbar=self.pbar))
        return features

    def fit(self, x, y):
        features = self._generate_features(x)
        self.regressor.fit(features, y)

    def predict(self, x):
        features = self._generate_features(x)
        return self.regressor.predict(features)