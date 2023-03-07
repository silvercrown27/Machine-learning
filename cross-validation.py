import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

path = r"datasets/home-data-for-ml-course/train.csv"
data = pd.read_csv(path).copy()

y = data.SalePrice
data_X = data.drop(['SalePrice'], axis=1)

num_cols = [col for col in data_X.columns if data_X[col].dtype in ['int64', 'float64'] and not
            data_X[col].isnull().any()]

categotical_cols = [col for col in data_X.columns if data_X[col].nunique() < 10 and
                    data_X[col].dtype == 'object']

my_cols = categotical_cols + num_cols
X = data_X[my_cols]

numerical_transformer = SimpleImputer(strategy='constant')
categotical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categotical_transformer, categotical_cols)
    ])

model = RandomForestRegressor(random_state=101, n_estimators=300, verbose=2)

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

cross_score = -1 * cross_val_score(my_pipeline, X, y,
                              cv=7, scoring='neg_mean_absolute_error')
print(cross_score.mean())