from sklearn.model_selection import cross_val_score
from category_encoders import MEstimateEncoder
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10
)
warnings.filterwarnings('ignore')

def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()

    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

path = f"datasets/FE-Course-Data/ames.csv"
df = pd.read_csv(path)

df.select_dtypes(["object"]).nunique()

df["SaleType"].value_counts()

x_encode = df.sample(frac=0.20, random_state=0)
y_encode = x_encode.pop("SalePrice")

x_pretrain = df.drop(x_encode.index)
y_train = x_pretrain.pop("SalePrice")
columns = ["Neighborhood", "MSSubClass"]

encoder = MEstimateEncoder(cols=columns[1], m=5.0)
encoder.fit(x_encode, y_encode)

x_train = encoder.transform(x_pretrain, y_train)

features = encoder.cols

plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(x_train[features], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
plt.show()

X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(x_train, y_train)

print(f"Baseline Score: {score_base} RMSLE")
print(f"Score With Encoding: {score_new:.4f} RMSLE")

# m = 0
# X_new = df.copy()
# y_new = df.pop("SalePrice")
#
# X_new['Count'] = range(len(X_new))
# X_new['Count'][1] = 0
#
# n_encoder = MEstimateEncoder(cols="Count", m=m)
# X = encoder.fit_transform(X_new, y_new )
# new_score =  score_dataset(X, y)
# print(f"Score: {new_score:.4f} RMSLE")
#
# plt.figure(dpi=90)
# ax = sns.distplot(y, kde=True, hist=False)
# ax = sns.distplot(X["Count"], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
# ax.set_xlabel("SalePrice")
# plt.show()