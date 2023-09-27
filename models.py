from pandas import Series, infer_freq
from numpy import arange
from statsmodels.tsa.statespace.sarimax import SARIMAX
from mlforecast import MLForecast
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from statsforecast.models import (
    DynamicOptimizedTheta,
    HistoricAverage,
)
from neuralforecast.models import GRU


def build_models(train: Series, horizon: int, seasonal_period: int) -> list:
    """Build models that will be used for the learning curves.

    Args:
        train (Series): The train set.
        horizon (int): The forecasting horizon.
        seasonal_period (int): The seasonal period of the datas.

    Returns:
        list: The list of all models.
    """
    freq = train.index.freq
    print(train)
    print(train.isna().sum())
    print(freq)
    return [
        SARIMAX(
            endog=train,
            order=(4, 1, 0),
            seasonal_order=(1, 1, 0, seasonal_period),
            freq=freq,
        ),
        MLForecast(
            models=[Ridge(), SVR(), GradientBoostingRegressor()],
            lags=arange(1, int(seasonal_period / 2), 1),
            freq=freq,
        ),
        StatsForecast(
            models=[DynamicOptimizedTheta(), HistoricAverage()],
            freq=freq,
        ),
        NeuralForecast(
            models=[GRU(input_size=2 * horizon, h=horizon, max_steps=20)],
            freq=freq,
        ),
    ]
