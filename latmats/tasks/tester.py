import numpy as np
import tqdm

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, \
    LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from latmats.tasks.loader import load_e_form, load_bandgaps, load_steels, \
    load_zT


def rmse(y_true, y_pred):
    """
    Root mean squared error.

    Args:
        y_true:
        y_pred:

    Returns:
        RMSE

    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


class AlgorithmBenchmark:
    def __init__(self, estimator, random_state_cv=1999, random_state_test=2001):

        config = {}
        self.estimator = estimator

        for problem in ["zT", "steels", "bandgap", "e_form"]:

            if problem == "zT":
                df = load_zT(all_data=False)
                cv_scheme = LeaveOneOut()
            elif problem == "steels":
                df = load_steels()
                cv_scheme = LeaveOneOut()
            elif problem == "bandgap":
                df = load_bandgaps()
                cv_scheme = KFold(n_splits=10, shuffle=True, random_state=random_state_cv)
            elif problem == "e_form":
                df = load_e_form()
                cv_scheme = KFold(n_splits=10, shuffle=True, random_state=random_state_cv)
            df_train, df_test = train_test_split(df, shuffle=True, random_state=random_state_test)

            problem_config = {
                "df_all": df,

                # Used for holding data related to cross validation for purposes
                # of architecture changes, hyperparameter tuning, etc.
                "cross_validation": {
                    "df": df_train,
                    "dfs_folds": None,  # as a list of validation predicted dfs
                    "cv_scores": None,  # as "metric": score pairs
                    "cv_scheme": cv_scheme,
                },

                # Used for generating a final score
                # Model is retrained on all the data available
                "testing": {
                    "df_train": df_train,
                    "df_test": df_test,
                    "final_estimator": None,
                    "df_predicted": None,
                    "scores": None,  # as "metric": score pairs
                }
            }
            config[problem] = problem_config
        self.data = config
        self.scorers = {
            "r2": r2_score,
            "mae": mean_absolute_error,
            "rmse": rmse,
            "explained_variance": explained_variance_score
        }

    @staticmethod
    def _get_target_from_df(df):
        cols = [col for col in df.columns if col != "composition"]
        if len(cols) != 1:
            raise ValueError(
                "The dataframe must have exactly one composition column (with "
                "column label 'composition') and one target column."
            )
        return cols[0]


    def cross_validate(self, problem, quiet=False):
        desc = f"cross validation for {self.estimator.__class__.__name__} on problem: '{problem}'"
        if not quiet:
            print(f"started: {desc}")
        splitter = self.data[problem]["cross_validation"]["cv_scheme"]
        df = self.data[problem]["cross_validation"]["df"]
        target = self._get_target_from_df(df)
        target_predicted = f"predicted {target}"

        scores = {s: [] for s in self.scorers.keys()}
        dfs_folds = []

        if quiet:
            splits = splitter.split(df)
        else:
            splits = tqdm.tqdm(splitter.split(df), desc=desc)

        for train_ix, test_ix in splits:
            df_train = df.iloc[train_ix]
            x_train = df_train.drop(labels=[target], axis=1)
            y_train = df_train[target]
            df_val = df.iloc[test_ix]
            x_val = df_val.drop(labels=[target], axis=1)
            y_val = df_val[target]
            self.estimator.fit(x_train, y_train)
            y_pred = self.estimator.predict(x_val)
            df_val[target_predicted] = y_pred
            dfs_folds.append(df_val)

            for scorer_name, scorer_fn in self.scorers.items():
                score = scorer_fn(y_val, y_pred)
                scores[scorer_name].append(score)
        scores_avg = {s: np.mean(scorelist) for s, scorelist in scores.items()}
        self.data[problem]["cross_validation"]["cv_scores"] = scores_avg
        self.data[problem]["cross_validation"]["dfs_folds"] = dfs_folds

        if not quiet:
            print(f"completed: {desc}")
        results = self.data[problem]["cross_validation"]
        return results

    def cross_validate_all_problems(self, return_summary_metrics=("mae",)*4, quiet=False):
        for problem in self.data.keys():
            self.cross_validate(problem, quiet=quiet)

        scores = {}
        for i, problem in enumerate(list(self.data.keys())):
            metric = return_summary_metrics[i]
            score = self.data[problem]["cross_validation"]["cv_scores"][metric]
            scores[problem] = score
            if not quiet:
                print(f"'{problem}' {metric}: {score}")
        return scores

    def test(self, problem):
        df_train = self.data[problem]["testing"]["df_train"]
        df_test = self.data[problem]["testing"]["df_test"]
        target = self._get_target_from_df(df)
        target_predicted = f"predicted {target}"

        x_train = df_train.drop(labels=[target], axis=1)
        y_train = df_train[target]
        x_test = df_test.drop(labels=[target], axis=1)
        y_test = df_test[target]

        self.estimator.fit(x_train, y_train)
        y_pred = self.estimator.predict(x_test)

        scores = {}
        for scorer_name, scorer_fn in self.scorers.items():
            score = scorer_fn(y_test, y_pred)
            scores[scorer_name] = score
        self.data[problem]["testing"]["scores"] = scores




from sklearn.ensemble import RandomForestRegressor

from abc import abstractmethod
from typing import Iterable

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

    def __init__(self):
        self.regressor = RandomForestRegressor(n_estimators=1000)
        self.stc = StrToComposition()
        ep = ElementProperty.from_preset("magpie")
        ef = ElementFraction()
        self.featurizer = MultipleFeaturizer([ep, ef])

    def _generate_features(self, x):
        comps = [o[0] for o in self.stc.featurize_many(x)]
        features = np.asarray(self.featurizer.featurize_many(comps))
        return features

    def fit(self, x, y):
        features = self._generate_features(x)
        self.regressor.fit(features, y)

    def predict(self, x):
        features = self._generate_features(x)
        return self.regressor.predict(features)



if __name__ == "__main__":

    # dummy_benchmark = AlgorithmBenchmark(DummyEstimator())
    # dummy_benchmark.cross_validate_all_problems(quiet=False)

    dummy_benchmark = AlgorithmBenchmark(RFEstimator())
    dummy_benchmark.cross_validate_all_problems(quiet=False)

