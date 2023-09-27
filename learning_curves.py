# Arrays and maths
import numpy as np
import pandas as pd

# Plottings tools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from mycolorpy import colorlist as mcp
from aquarel import load_theme

# Modelling
from mlforecast import MLForecast
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Allow overloading
from multipledispatch import dispatch

# Get parameters in CLI
from argparse import ArgumentParser

# Loading models
from models import build_models

# Plotting settings
theme = load_theme("scientific")
theme.apply()


def __kalman_imputer(datas: pd.Series, seasonal_period: int) -> pd.Series:
    """Impute the missing values of a pd.Series using the Kalman Filter.

    Args:
        data (pd.Series): The time serie to impute.
        seasonal_period (int): The seasonal period of the datas.

    Returns:
        (pd.Series) : The imputed serie.
    """
    missing_indexes = datas[datas.apply(np.isnan)].index

    imputer = SARIMAX(
        endog=datas,
        order=(0, 1, 0),
        seasonal_order=(0, 1, 0, seasonal_period),
        enforce_invertibility=False,
        enforce_stationarity=False,
        freq=pd.infer_freq(datas.index),
    )

    imputer_fitted = imputer.fit()
    recon = imputer_fitted.predict()

    datas.loc[datas.index.isin(missing_indexes)] = recon.loc[
        recon.index.isin(missing_indexes)
    ]
    return datas


def load_datas(
    path_to_load: str,
    horizon: int,
    index_name: str = "DATE",
    seasonal_period: int = 12,
) -> [pd.Series, pd.Series]:
    """The the csv file containing the datas.

    Args:
        path_to_load (str): The path containing the csv file.
        horizon (int): The forecasting horizon targeted.
        index_name (str, optional): The name of the date columns to use as an index. Defaults to "DATE".
        seasonal_period (int): The seasonal period of the datas, used to impute the missing values.

    Returns:
        pd.Series: The training set.
        pd.Series: The validation set.
    """
    datas = pd.read_csv(path_to_load, index_col=index_name)
    datas.index = pd.to_datetime(datas.index)
    datas = datas.squeeze()
    datas = datas.asfreq(pd.infer_freq(datas.index))
    datas = datas.astype(float)
    datas = __kalman_imputer(datas, seasonal_period)
    return datas.iloc[:-horizon], datas.iloc[-horizon:]


def __nixtla_preprocess_column(serie: pd.Series) -> pd.DataFrame:
    """Convert a pandas Series into a DataFrame usable for Nixtla models.

    Args:
        serie (pd.Series): The Series to convert.

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    returned_dataset = pd.DataFrame()
    returned_dataset["unique_id"] = [serie.name] * len(serie)
    returned_dataset["ds"] = serie.index
    returned_dataset["y"] = serie.values
    return returned_dataset


def split_chunks(data: np.array, n_chunks: int = 15) -> [list, list]:
    """Split an iterable into n_chunks equal chunks.

    Args:
        data (np.array): The data to split.
        n_chunks (int, optional): The number of chunks that you want. Defaults to 15.

    Returns:
        list: The list of all the chunks.
        list: The list of the indexes to split to get the chunks.
    """
    chunks = np.array_split(ary=data, indices_or_sections=n_chunks)
    return chunks, [min(chunk.index) for chunk in chunks]


@dispatch(datas=pd.Series, horizon=int, model=MLForecast)
def fit_forecast(datas: pd.Series, horizon: int, model: MLForecast) -> dict:
    """Fit the models contained in the MLForecast object and forecast the horizon.

    Args:
        datas (pd.Series): The training set to fit the models on.
        horizon (int): The forecasting horizon.
        model (MLForecast): The MLForecast object.

    Returns:
        dict: The dictionnary containing the forecasts of the models contained in the MLForecast object, with the following structure : {model : forecast}
    """
    fitted = model.fit(__nixtla_preprocess_column(datas))
    pred = fitted.predict(horizon)
    pred = pred.set_index("ds", drop=True)

    return {
        model: pred[model]
        for model in [x for x in pred.columns if x not in ["unique_id", "ds", "y"]]
    }


@dispatch(datas=pd.Series, horizon=int, model=StatsForecast)
def fit_forecast(datas: pd.Series, horizon: int, model: StatsForecast) -> dict:
    """Fit the models contained in the StatsForecast object and forecast the horizon.

    Args:
        datas (pd.Series): The training set to fit the models on.
        horizon (int): The forecasting horizon.
        model (MLForecast): The StatsForecast object.

    Returns:
        dict: The dictionnary containing the forecasts of the models contained in the StatsForecast object, with the following structure : {model : forecast}
    """
    fitted = model.fit(__nixtla_preprocess_column(datas))
    pred = fitted.forecast(h=horizon)
    pred = pred.set_index("ds", drop=True)

    return {
        model: pred[model]
        for model in [x for x in pred.columns if x not in ["unique_id", "ds", "y"]]
    }


@dispatch(datas=pd.Series, horizon=int, model=NeuralForecast)
def fit_forecast(datas: pd.Series, horizon: int, model: NeuralForecast) -> dict:
    """Fit the models contained in the StatsForecast object and forecast the horizon.

    Args:
        datas (pd.Series): The training set to fit the models on.
        horizon (int): Not used, here to respect the overload syntax. The horizon is specified in the NeuralForecast object.
        model (MLForecast): The NeuralForecast object.

    Returns:
        dict: The dictionnary containing the forecasts of the models contained in the NeuralForecast object, with the following structure : {model : forecast}
    """
    model.fit(__nixtla_preprocess_column(datas))
    pred = model.predict()
    pred = pred.set_index("ds", drop=True)

    return {
        model: pred[model]
        for model in [x for x in pred.columns if x not in ["unique_id", "ds", "y"]]
    }


@dispatch(datas=pd.Series, horizon=int, model=SARIMAX)
def fit_forecast(datas: pd.Series, horizon: int, model: SARIMAX) -> dict:
    """Fit the SARIMAX and forecast the horizon.

    Args:
        datas (pd.Series): Not used, here to respect the overload syntax. Fitting datas must be specified when defining the SARIMAX (the 'endog' parameter).
        horizon (int): The forecasting horizon.
        model (MLForecast): The SARIMAX object.

    Returns:
        dict: The dictionnary containing the forecast of the SARIMAX, with the following structure : {model : forecast}
    """
    fitted = model.fit()
    pred = fitted.forecast(horizon)
    return {"SARIMAX": pred}


def compute_validation_chunks(
    train_set: pd.Series, slice_indices: list, validation_set: pd.Series, models: list
) -> dict:
    """Fit the models on differents history lengths, and compute the forecast performance (MAE) for each chunks.

    Args:
        train_set (pd.Series): The train set.
        slice_indices (list): The slice indexes to split the chunks at each iteration.
        validation_set (pd.Series): The validation set used to compute forecast performance.
        models (list): The list of models to evaluate (using the fit_forecast method).

    Returns:
        dict: The performance dictionnary for each model.
    """
    for iteration, slice_idx in enumerate(np.flip(slice_indices)):
        sliced_train = train_set.loc[slice_idx:]

        for sub_iteration, model in enumerate(models):
            if sub_iteration == 0:
                fcst_dict = fit_forecast(sliced_train, validation_set.shape[0], model)
            else:
                fcst_dict = fcst_dict | fit_forecast(
                    sliced_train, validation_set.shape[0], model
                )

        if iteration == 0:  # Initiate the perf on validation set dictionnary
            val_perfs = {
                model: [np.mean(np.abs(fcst_dict[model] - validation_set))]
                for model in fcst_dict.keys()
            }

        else:  # Update the perf on validation set dictionnary
            for model in val_perfs.keys():
                val_perfs[model].append(
                    np.mean(np.abs(fcst_dict[model] - validation_set))
                )

    return val_perfs


def plot_validation_curves(
    chunks: list, val_perfs: dict, validation_set: pd.Series, path_to_save: str
) -> None:
    """Plot the validation curves and the chunked dataset

    Args:
        chunks (list): The list containing the chunks of the dataset.
        val_perfs (dict): The performance of the model for each chunks.
        validation_set (pd.Series): The validation set, used to perform forecast performance.
        path_to_save (str): The path where to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
    markers = list(Line2D.markers.keys())

    ax1.set_title("Learning Curves", fontweight="bold", pad=5)
    ax1.set_xlabel("History size", color="black")
    ax1.set_ylabel(
        "Mean Absolute Error (MAE)", color="black", labelpad=10, fontweight="bold"
    )
    for iteration, model in enumerate(val_perfs.keys()):
        ax1.plot(
            np.cumsum([chunk.shape[0] for chunk in chunks]),
            val_perfs[model],
            label=f"{model}",
            marker=markers[iteration],
        )
    ax1.legend(loc="upper right")

    # 2nd plot
    ax2.set_title("Chunked Dataset", fontweight="bold", pad=5)
    ax2.set_xlabel("Date", color="black")
    ax2.set_ylabel("Value", color="black", labelpad=10, fontweight="bold")

    n_chunks = len(chunks)

    if n_chunks < 10:
        colors = plt.cm.Set1
        cmap = [colors(i) for i in range(n_chunks)]

    else:
        cmap = mcp.gen_color(cmap="winter", n=n_chunks)

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx > 0:
            ax2.axvline(x=chunk.index[0], color="grey", linestyle="--")
            ax2.plot(chunk, color=cmap[chunk_idx])
            ax2.fill_between(
                chunk.index,
                np.float32(chunk),
                [0] * chunk.shape[0],
                color=cmap[chunk_idx],
                alpha=0.3,
            )
        else:
            ax2.fill_between(
                chunk.index,
                np.float32(chunk),
                [0] * chunk.shape[0],
                color=cmap[chunk_idx],
                alpha=0.3,
                label="Chunked Train set",
            )

            ax2.plot(chunk, color=cmap[chunk_idx])

    ax2.plot(validation_set, color="salmon")
    ax2.fill_between(
        validation_set.index,
        np.float32(validation_set),
        [0] * validation_set.shape[0],
        color="salmon",
        alpha=0.3,
        label="Test set",
    )

    ax2.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(path_to_save, bbox_inches="tight")


def main():
    parser = ArgumentParser(description="Get the needed hyperparameters")

    parser.add_argument("--read_path", type=str, help="Path to the csv file to read")

    parser.add_argument(
        "--plot_name", type=str, help="Name of the plot (with the image extension)"
    )

    parser.add_argument(
        "--index_name",
        type=str,
        help="The name of the date column to use as index",
        default="DATE",
    )

    parser.add_argument(
        "--seasonal_period",
        type=int,
        help="The seasonal period of the datas",
    )

    parser.add_argument(
        "--forecast_horizon",
        type=int,
        help="Forecasting horizon to evaluate the models on",
    )

    parser.add_argument(
        "--n_chunks", type=int, help="The number of chunks to do", default=15
    )

    args = parser.parse_args()

    read_path = args.read_path
    plot_name = args.plot_name
    index_name = args.index_name
    seasonal_period = args.seasonal_period
    forecast_horizon = args.forecast_horizon
    n_chunks = args.n_chunks

    train, validation = load_datas(
        read_path,
        forecast_horizon,
        index_name,
        seasonal_period,
    )
    print("Data load ok")

    chunks, slice_indices = split_chunks(train, n_chunks=n_chunks)
    print("Chunks ok")

    val_perfs = compute_validation_chunks(
        train,
        slice_indices,
        validation,
        build_models(train, forecast_horizon, seasonal_period),
    )
    print("Validation process ok")

    plot_validation_curves(
        chunks, val_perfs, validation, path_to_save=f"plots/{plot_name}"
    )
    print("Plotting curves ok")


if __name__ == "__main__":
    main()
