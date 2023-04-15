import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from time_utils import plot_periodogram, seasonal_plot

simplefilter('ignore')

plt.style.use('seaborn-whitegrid')
plt.rc("figure", autolayout=True, figsize=(14, 6))
plt.rc("axes", titleweight="bold", titlesize="large",
       labelweight="bold", labelsize=16, labelpad=10)

plot_params = dict(color="0.75", style=".-", legend=False,
                   markeredgecolor="0.25", markerfacecolor="0.25")

path_1 = f"C://Datasets/store-sales-time-series-forecasting/holidays_events.csv"
path_2 = f"C://Datasets/store-sales-time-series-forecasting/train.csv"
dtype1 = {
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
        }
dtype2 = {
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    }

holiday_events = pd.read_csv(path_1, dtype=dtype1, parse_dates=['date'], infer_datetime_format=True)
holiday_events = holiday_events.set_index('date').to_period('D')

store_sales = pd.read_csv(path_2, usecols=['store_nbr', 'family', 'date', 'sales'],
                          dtype=dtype2, parse_dates=['date'], infer_datetime_format=True)

store_sales['date'] = store_sales.date.dt.to_period("D")
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales.groupby('date').mean().squeeze().loc['2017']
)
print(average_sales.head())


X = pd.DataFrame(average_sales)
X["week"] = X.index.week
X['day'] = X.index.dayofweek
y = X.sales.T
plot_periodogram(average_sales)
seasonal_plot(X, y=y, period='week', freq='day')
plt.show()
