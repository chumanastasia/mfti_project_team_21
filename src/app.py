import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from src.extractor import TSExtractor, ConfigExtractor, TrainExtractor
from src.train import LinerTrainer
from src.config import Settings


APP_TITTLE = "🛢️ Мониторинг показателя плотности нефти".title()
MODEL_NAME = "MODEL_dRo"

_settings = Settings()
_ts_extractor = TSExtractor(dsn=str(_settings.postgres_dsn))
_config_extractor = ConfigExtractor(dsn=str(_settings.postgres_dsn))
_train_extractor = TrainExtractor(dsn=str(_settings.postgres_dsn))

st.set_page_config(
    page_title=APP_TITTLE,
    page_icon="🧊",
)


@st.cache_data(ttl=60 * 60 * 24)
def get_data():
    """Load and cache the data"""

    _config = _config_extractor.get_config_by_name(MODEL_NAME)

    set_ = _train_extractor.extract_train_set_by_model_name(MODEL_NAME)
    line_trainer = LinerTrainer(set_.iloc[:, :-2], set_.iloc[:, -1])
    model = line_trainer.train()
    ts = line_trainer.predict(model)
    ts.columns = [MODEL_NAME]
    ts = pd.concat([ts, set_.iloc[:, -1]], axis=1)
    return ts.iloc[500:, :]


st.title(APP_TITTLE)
st.write("Система мониторинга показателя качества нефтепродуктов")
source = get_data()

r2, rmse = st.columns(2)

r2.metric("R2", round(r2_score(source.iloc[:, 1], source.iloc[:, 0]), 2))
rmse.metric("RMSE", round(mean_squared_error(source.iloc[:, 0], source.iloc[:, 1], squared=False), 2))


st.line_chart(source, color=["#4CB9E7", "#FFECD6"])

st.write("## Исходный Код")

st.write(
    "Можно посмотреть на [GitHub]"
    "(https://github.com/chumanastasia/mfti_project_team_21)"
)

