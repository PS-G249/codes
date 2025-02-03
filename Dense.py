import numpy as np
class Dense:
    def __init__(self, input_dim, output_dim):
        np.random.seed(42)
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.biases = np.zeros(output_dim)

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights).round(decimals=3) + self.biases

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(self.input_data.T, d_output)
        d_biases = np.sum(d_output, axis=0)
        d_input = np.dot(d_output, self.weights.T)
        self.weights -= (learning_rate * d_weights)
        self.biases -= (learning_rate * d_biases)
        return d_input