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
y_train = (train_data[0]).reshape(1000, 1)
X_valid = scaler.transform(train_data[1:n])
y_valid = valid_data[0]

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

# The Neural Network

class NeuralNetwork:
    def __init__(self, layer_density=556, learning_rate=0.001, epochs=200):
        self.epochs = epochs
        self.hidden_size = layer_density
        self.input_size = X_train.shape[0]
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = softmax(self.z2)

    def backward(self, X, y):
        y = self.y.astype(int)

        # Backward pass
        delta3 = self.y_hat
        delta3[range(len(X)), y] -= 1
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def predict(self, X):
        # Predict the class of input data
        self.forward(X)
        return np.argmax(self.y_hat, axis=1)

    def loss(self, X, y):
        # Calculate the cross-entropy loss
        self.forward(X)
        m = len(X)
        loss = -np.sum(np.log(self.y_hat[range(m), y])) / m
        return loss

    def accuracy(self, X, y):
        # Calculate the accuracy
        y_pred = self.predict(X.T)
        return np.mean(y_pred == y.T)

    def fit(self, X, y, verbose=False):
        self.X = X.T
        self.y = y.T
        for epoch in range(self.epochs + 1):
            self.forward(self.X)
            self.backward(self.X, self.y)
            ls = self.loss(self.X, self.y)
            accuracy = self.accuracy(X, y)
            if (epoch % 10 == 0 & verbose):
                print("Epoch " + "=" * 25 + ">: " + f"{epoch}")
                print(f"Loss: {ls}")
                print(f"Accuracy: {accuracy}")
                print("\n")

            if (np.argwhere(np.isnan(self.b1))).any():
                break


model = NeuralNetwork()
model.fit(X_train, y_train, verbose=True)
acc = model.accuracy(X_valid, y_valid)
print(f"Accuracy: {acc}")
