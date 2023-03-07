import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score

path = r"C:\Users\brada\Downloads\python programs\machinelearning\datasets\Covid Data.csv"
data = pd.read_csv(path)

pd.set_option('display.max_columns', 21)
pd.set_option('display.width', 2000)

categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
targets = ['CLASIFFICATION_FINAL', 'ICU']

print(data.columns)
cols = categorical_cols + targets
y = data[targets]
X = data.drop(cols, axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=99)

my_model = XGBRegressor(random_state=10, n_estimators=50)
cross_score = cross_val_score(my_model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error', verbose=2)

print(-1 * cross_score)
print(-1 * cross_score.mean())