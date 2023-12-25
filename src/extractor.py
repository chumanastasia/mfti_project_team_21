from src.repo import TSRepository
from pydantic import PostgresDsn
import pandas as pd
from src.schemas import ModelConfig


class ConfigExtractor:
    """A class to extract configuration from the database."""

    def __init__(self, dsn: PostgresDsn) -> None:
        """Initialize a connection to the database."""

        self._repo = TSRepository(dsn=dsn)

    def get_config_by_name(self, model_name: str) -> ModelConfig:
        """Get the configuration of a model by its name."""
        query = "SELECT model_name, config FROM model_configs WHERE model_name = %s"
        model_conf = self._repo.select_one(query=query, values=(model_name,))
        return ModelConfig(**model_conf[-1])


class TSExtractor:
    """A class to extract time series from the database."""

    def __init__(self, dsn: PostgresDsn) -> None:
        """Initialize a connection to the database."""

        self._repo = TSRepository(dsn=dsn)

    def get_ts_by_name(self, ts_name: str) -> pd.DataFrame:
        """Get the time series by its name."""
        query = "SELECT time, value FROM time_series WHERE tag = %s"
        ts = pd.read_sql_query(query, self._repo.get_connection(), params=(ts_name,))

        ts.drop_duplicates("time", inplace=True)
        ts.rename(columns={"value": ts_name}, inplace=True)
        ts.set_index("time", inplace=True)
        ts.sort_index(inplace=True)
        return self._process_ts(ts)

    def get_ts_by_names(self, ts_names: list[str]) -> dict[str, pd.DataFrame]:
        """Get the time series by its name."""
        tss = {}
        for tag in ts_names:
            tss[tag] = self.get_ts_by_name(tag)

        return tss

    @classmethod
    def _process_ts(cls, ts: pd.DataFrame) -> pd.DataFrame:
        Q1 = ts.quantile(0.25)
        Q3 = ts.quantile(0.75)
        IQR = Q3 - Q1

        return ts[~((ts < (Q1 - 1.5 * ts)) | (ts > (Q3 + 1.5 * IQR))).any(axis=1)]


class TrainExtractor:
    """A class to extract training data from the database."""

    def __init__(self, dsn: PostgresDsn) -> None:
        """Initialize a connection to the database."""

        self._repo = TSRepository(dsn=dsn)
        self._conf_extr = ConfigExtractor(dsn=dsn)
        self._ts_extr = TSExtractor(dsn=dsn)

    def extract_train_set_by_model_name(self, model_name: str) -> pd.DataFrame:
        """Extract the regressors of a model by its name."""
        config = self._conf_extr.get_config_by_name(model_name=model_name)
        regressors = config.inputs_tags
        teacher_tag = config.teacher_tag

        teacher = self._ts_extr.get_ts_by_name(teacher_tag)
        X = pd.DataFrame(index=teacher.index)

        regressors_ts = self._ts_extr.get_ts_by_names(regressors)

        for regressor in regressors:
            ts = regressors_ts[regressor].reindex(teacher.index, method="ffill")

            X = pd.concat([X, ts], axis=1)

        X = pd.concat([X, teacher], axis=1)
        return X.dropna()
