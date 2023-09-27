# Learning-curves-TSA
Learning curves for time series models that allow to see the variation of the forecasting performance with respect to the historic length used to fit the models.

Data used : https://www.kaggle.com/datasets/shenba/time-series-datasets?resource=download
Commands used to get the examples plot :<br><br>

  python learning_curves.py --read_path "datas/elec_prod.csv" --forecast_horizon 36 --n_chunks 15 --plot_name "elec_prod.svg" --seasonal_period 12
<br><br>
  python learning_curves.py --read_path "datas/daily_temp.csv" --forecast_horizon 40 --n_chunks 20 --plot_name "daily_temp.svg" --seasonal_period 30 --index_name "Date"
