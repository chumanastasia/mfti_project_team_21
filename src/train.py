import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge


class LinerTrainer:
    """A class to train a model."""

    _lasso = Lasso
    _linear = LinearRegression
    _ridge = Ridge

    def __init__(self, regressors: pd.DataFrame, teacher: pd.DataFrame) -> None:
        """Initialize the trainer."""

        self._regressors = regressors
        self._teacher = teacher

    def train(self):
        """Train the model."""

        model = self._lasso(alpha=10 ** -7)
        model.fit(self._regressors, self._teacher)

        return model

    def predict(self, model) -> pd.DataFrame:
        """Predict the teacher."""

        return pd.DataFrame(model.predict(self._regressors), index=self._teacher.index, columns=[self._teacher.name])
