import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pmdarima as pm
from pmdarima.datasets.stocks import load_msft
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape


test_df = pd.read_csv('./hw3_Data2/test.csv')
train_df = pd.read_csv('./hw3_Data2/train.csv')
df = pd.concat([train_df, test_df])

y_train = train_df['Close'].values
y_test = test_df['Close'].values

from pmdarima.arima import ndiffs

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")

model = pm.ARIMA(order = (0, n_diffs, 0))
model.fit(y_train)
 
def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

forecasts = []
confidence_intervals = []

for new_ob in y_test:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)
    
    # Updates the existing model with a small number of MLE steps
    model.update(fc)
    
print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
print(f"SMAPE: {smape(y_test, forecasts)}")

x = np.arange(df.shape[0])

figure(figsize=(8, 4), dpi=100)
#plt.plot(x[:train_df.shape[0]], y_train, color='blue', label='Training Data')
#plt.plot(x[train_df.shape[0]:], y_test, color='red', label='Actual Price')
plt.plot(df['Close'].values, color='blue', label='The Whole Stock')
plt.plot(x[train_df.shape[0]:], forecasts, color='green',label='Forecast Data ')

plt.title('Taiwan’s Stock Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()
plt.show()