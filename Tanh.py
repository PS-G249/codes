import numpy as np

class Tanh:
    def __init__(self):
        self.output = None  # To store the output for backward pass

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, gradient, learning_rate):
        return gradient * (1 - self.output ** 2)
