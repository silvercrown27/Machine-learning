import pandas as pd
import numpy as np

path = f"C://Datasets/digit-set/digit-recognizer/train.csv"
data = pd.read_csv(path)

pd.set_option('display.width', 2000)

data = data.T

print(data.head())

train_X = data.iloc[1:, :10000]
train_y = data.iloc[0, :10000]
valid_X = data.iloc[1:, 10001:]
valid_y = data.iloc[0, 10001:]

def relu(x):
    return max(0, x)

def deriv_relu(x):
    return x < 0

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True))

class NeuralNetwork:
    def __init__(self, layer_density=10, learning_rate=0.3, epochs=1000):
        self.epochs = epochs
        self.hidden_size = layer_density
        self.input_size = train_X.shape[0]
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases1 = np.random.randn(self.hidden_size, 1)
        self.weights2 = np.random.randn(self.hidden_size, self.hidden_size)
        self.biases2 = np.random.randn(self.hidden_size, 1)

    def forward_prop(self, train_X):
        z1 = np.dot(train_X, self.weights1) + self.biases1
        a1 = relu(z1)
        z2 = np.dot(a1, self.weights2) + self.biases2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    def back_prop(self, train_X, train_y, z1, a1, z2, a2):
        dz2 = (a2 - train_y)
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(db2, self.weights2.T)
        dz1 = da1 * deriv_relu(z1)
        dw1 = np.dot(train_X, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return dw1, db1, dw2, db2

    def update_params(self, dw1, db1, dw2, db2):
        self.weights2 -= self.learning_rate * dw2
        self.biases2 -= self.learning_rate * db2
        self.weights1 -= self.learning_rate * dw1
        self.weights1 -= self.learning_rate * db1

    def fit(self, X, y):
        for i in range(self.epochs):
            params = self.forward_prop(X)
            params_2 = self.back_prop(X, y, params)
            self.update_params(params_2)
            if i % 100 == 0:
                print("Epoch" + "=" * 20 + ">: " + f"{i}")


NeuralNetwork().fit(train_X, train_y)