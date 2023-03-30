import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

boston = datasets.load_boston(return_X_y=False)

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=99)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

print('coefficients: \n', reg.coef_)
print(f"Variance Score: {reg.score(x_test, y_test)}")

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color="green", s=10, label='Train data')
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color="blue", s=10, label='Test data')

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual Errors")
plt.show()


