import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess

plt.style.use('seaborn-whitegrid')
plt.rc("figure", autolayout=True, figsize=(16, 8))
plt.rc("axes", titleweight='bold', titlesize='large',
       labelweight='bold', labelsize='large', titlepad=10)

plot_params = dict(
    color="0.75", style=".-", legend=False,
    markeredgecolor="0.25", markerfacecolor="0.25"
)

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 20)

path_1 = f"datasets/ts-course-data/us-retail-sales.csv"
path_2 = f"datasets/store-sales-time-series-forecasting/train.csv"

retail_sales = pd.read_csv(path_1, parse_dates=['Month'],
                           index_col='Month').to_period("D")
food_sales = retail_sales.loc[:, 'FoodAndBeverage']
auto_sales = retail_sales.loc[:, 'Automobiles']

dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(path_2, dtype=dtype,
                          parse_dates=['date'], infer_datetime_format=True)
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']

print(store_sales.head())
print(retail_sales.head())
print(average_sales)

#  12-day data for food_sales
# trend = food_sales.rolling(
#     window=12, center=True, min_periods=6
# ).mean()
#
# ax = food_sales.plot(**plot_params)
# ax = trend.plot(ax=ax, linewidth=3)


#  yearly trend(Average sales
# trend = average_sales.rolling(
#     window=365, center=True, min_periods=183
# ).mean()
# ax = average_sales.plot(**plot_params, alpha=0.5)
# ax = trend.plot(ax=ax, linewidth=3)

y = average_sales.copy()

dp = DeterministicProcess(index=y.index, order=11, drop=True)  # applying the constant option reduces accuracy
X = dp.in_sample()

X_fore = dp.out_of_sample(steps=90)
print(X.head())


model = LinearRegression()
model.fit(X, y)

preds = pd.Series(model.predict(X), index=X.index)
error = mean_absolute_error(y, preds)
print(f"Mean Absolute Error :\n{error}\n")

y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
print(y_fore[0:10])

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = preds.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend()

plt.show()
