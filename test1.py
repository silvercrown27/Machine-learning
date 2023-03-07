import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.biases2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.biases1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        delta2 = (self.a2 - y) * sigmoid_prime(self.z2)
        d_weights2 = np.dot(self.a1.T, delta2)
        d_biases2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.weights2.T) * sigmoid_prime(self.z1)
        d_weights1 = np.dot(X.T, delta1)
        d_biases1 = np.sum(delta1, axis=0)
        self.weights1 -= learning_rate * d_weights1
        self.biases1 -= learning_rate * d_biases1
        self.weights2 -= learning_rate * d_weights2
        self.biases2 -= learning_rate * d_biases2

    def train(self, X, y, learning_rate, num_epochs, ax=None):
        losses = []
        for epoch in range(num_epochs):
            self.forward(X)
            loss = np.mean(np.square(y - self.a2))
            losses.append(loss)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                if ax is not None:
                    ax.clear()
                    ax.set_xlim(0, num_epochs)
                    ax.set_ylim(0, max(losses))
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.plot(range(epoch + 1), losses)
        return losses


def animate_losses(losses):
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(losses))
    ax.set_ylim(0, max(losses))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [])

    def update(frame):
        line.set_data(range(frame + 1), losses[:frame + 1])
        return line,

    anim = FuncAnimation(fig, update, frames=len(losses), interval=50, blit=True)
    plt.show()


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

losses = nn.train(X, y, learning_rate=0.1, num_epochs=10000, ax=plt.gca())

animate_losses(losses)
