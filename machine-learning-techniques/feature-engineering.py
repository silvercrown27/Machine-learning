import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes",
       labelweight="bold",
       labelsize="large",
       titleweight="bold",
       titlesize=14,
       titlepad=10
       )

def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()

    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

path = r"datasets/FE-Course-Data/ames.csv"
df = pd.read_csv(path)

X = df.copy()
y = X.pop("SalePrice")

features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GrLivArea']

X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std()

kmeans = KMeans(random_state=0, max_iter=10, n_clusters=10, n_init=10)
X['Cluster'] = kmeans.fit_predict(X_scaled)

Xy = X.copy()
Xy['Cluster'] = Xy.Cluster.astype('category')
Xy['SalePrice'] = y

sns.relplot(
    x="value", y='SalePrice', hue="Cluster", col='variable',
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(value_vars=features, id_vars=['SalePrice', 'Cluster'])
    )

plt.show()
print(score_dataset(X, y))