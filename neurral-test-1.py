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

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True))

class NeuralNetwork:
    def __init__(self, train_X, hidden_size):
        self.input_size = train_X.shape[0]
        self.hidden_size = hidden_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases1 = np.random.randn(self.hidden_size, 1)
        self.weights2 = np.random.randn(self.hidden_size, self.hidden_size)
        self.biases2 = np.random.randn(self.hidden_size, 1)

    def forward_prop(self, train_X):
        self.z1 = np.dot(train_X, self.weights1) + self.biases1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.a2 = softmax(self.z2)
        return self.a2

    def back_propagation(self, train_X, train_y):
        delta2 = (self.a2 - train_y)
