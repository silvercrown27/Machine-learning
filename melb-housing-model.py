import numpy as np
import pandas as pd
from pandas import set_option
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

path = r'C:\Users\brada\Downloads\python programs\machinelearning\datasets\Melbourne_housing_FULL.csv'
data = pd.read_csv(path)

print(data.shape)
print(data.describe())
print(data.columns)

set_option('display.width', 100)
set_option('display.precision', 2)
filtered_data = data.dropna(axis=0)
feature_names = ['Rooms', 'Distance', 'Bedroom2', 'Landsize', 'BuildingArea']
X = filtered_data[feature_names]
y = filtered_data.Price
print(X.head())

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=101, shuffle=True)

def get_mae(max_leaf_node, train_X, test_X, train_y, test_y):
    model = DecisionTreeClassifier(random_state=101, max_leaf_nodes=max_leaf_node)
    model.fit(train_X, train_y)
    val_data = model.predict(test_X)
    mae = mean_absolute_error(test_y, val_data)
    return mae

tree_leaf_sizes = [10, 25, 50, 75, 100, 125, 150]

score = {}
for max_leaf_node in tree_leaf_sizes:
    error = get_mae(max_leaf_node, train_X, test_X, train_y, test_y)
    score[max_leaf_node] = error

best_tree_size = min(score, key=score.get)
print(score)
print(best_tree_size)

final_model = DecisionTreeClassifier(random_state=101, max_leaf_nodes=best_tree_size)
final_model.fit(X, y)
pred = final_model.predict(X)
error = mean_absolute_error(y, pred)
print(error)
accuracy = accuracy_score(y, pred)
print(accuracy)