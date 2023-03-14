import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
from sklearn.datasets import load_digits

digits = load_digits()

# Normalize the pixel values between 0 and 1
X = digits.images
print(X.shape)
y = digits.target
X = X / 16.0
print(X.shape)
print(y.shape)
# Reshape the input images into 1D arrays
X = X.reshape(X.shape[0], -1)
print(X.shape)
# One-hot encode the target labels
y_one_hot = np.zeros((y.shape[0], 10))
y_one_hot[np.arange(y.shape[0]), y] = 1
print(y_one_hot.shape)
# Define the neural network architecture
input_size = X.shape[1]
hidden_size = 128
output_size = 10
learning_rate = 0.1

# Initialize the weights
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


# Define the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)


# Define the softmax function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Train the neural network using mini-batch gradient descent
batch_size = 64
num_epochs = 10
num_batches = X.shape[0] // batch_size

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch in range(num_batches):
        # Select a random batch of inputs and labels
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y_one_hot[indices]

        # Forward propagation
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        print(A2)

        # Compute the loss
        loss = -np.sum(y_batch * np.log(A2)) / batch_size
        epoch_loss += loss

        # Backward propagation
        dZ2 = A2 - y_batch
        dW2 = np.dot(A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X_batch.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        # print(dW1, db1, dW2, db2)
        # Update the weights
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    epoch_loss /= num_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Plot the training accuracy
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
y_pred = np.argmax(Z2, axis=1)
accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.4f}")
