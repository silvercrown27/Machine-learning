import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

path = r"datasets/pima-indians-diabetes.csv"
headers = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(path, names=headers)
array = data.values
X = array[:, :8]
Y = array[:, 8]
'''
alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
param_grid = dict(alpha=alphas)

model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)

print(grid.best_score_)
print(grid.best_estimator_.alpha)
'''

#  randomized search

param_grid = {'alpha': uniform()}
model = Ridge()
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, random_state=7)
random_search.fit(X, Y)

print(random_search.best_score_)
print(random_search.best_estimator_)