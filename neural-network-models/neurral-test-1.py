import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

path = f"C://Datasets/digit-set/digit-recognizer/train.csv"
data = pd.read_csv(path)

data = np.array(data)

scaler = StandardScaler()

# Fit the scaler to the data and transform the data
m, n = data.shape
np.random.shuffle(data)

valid_data = data[0:10000].T
train_data = data[10000:11000].T

X_train = scaler.fit_transform(train_data[1:n])
y_train = (train_data[0, 0:1000]).reshape(1000, 1)
X_valid = scaler.transform(train_data[1:n])
y_valid = train_data[0, 1000:m]


def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return x > 0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

class NeuralNetwork:
    def __init__(self, layer_density=128, learning_rate=0.001, epochs=100):
        self.epochs = epochs
        self.hidden_size = layer_density
        self.input_size = X_train.shape[0]
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, 10)
        self.biases2 = np.zeros((1, 10))

    def forward_prop(self, X_train):
        z1 = np.dot(X_train.T, self.weights1) + self.biases1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.biases2
        a2 = z2
        return z1, a1, z2, a2

    def back_prop(self, X_train, y_train, z1, a1, a2):
        dz2 = (a2 - y_train)
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(self.weights2, dz2.T)
        dz1 = da1 * deriv_sigmoid(z1.T)
        dw1 = np.dot(X_train, dz1.T)
        db1 = np.sum(dz1.T, axis=0)
        return dw1, db1, dw2, db2

    def update_params(self, dw1, db1, dw2, db2):
        self.weights1 -= self.learning_rate * dw1
        self.biases1 -= self.learning_rate * db1
        self.weights2 -= self.learning_rate * dw2
        self.biases2 -= self.learning_rate * db2

    def predict(self, X, a2):
        # Predict the class of input data
        self.forward_prop(X)
        return np.argmax(a2, axis=1)

    def loss(self, X, y, a2):
        # Calculate the cross-entropy loss
        self.forward_prop(X)
        m = len(X)
        loss = -np.sum(np.log(a2[range(m), y])) / m
        return loss

    def accuracy(self, X, y, a2):
        # Calculate the accuracy
        y_pred = self.predict(X, a2)
        return np.mean(y_pred == y)

    def fit(self, X, y):
        min_error = []
        for epoch in range(self.epochs):
            z1, a1, z2, a2 = self.forward_prop(X)
            dw1, db1, dw2, db2 = self.back_prop(X, y, z1, a1, a2)
            self.update_params(dw1, db1, dw2, db2)
            accuracy = self.accuracy(X, y, a2)
            if epoch % 10 == 0:
                print("Epoch" + "=" * 25 + ">: " + f"{epoch}")
                print(f"Accuracy: {accuracy}")

            if (np.argwhere(np.isnan(self.biases1))).any():
                break

        print(min_error)


model = NeuralNetwork()
model.fit(X_train, y_train)