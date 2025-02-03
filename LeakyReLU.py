import numpy as np
class LeakyReLU:
    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(input_data,0.01*input_data)
    def backward(self, d_output,learning_rate):
        input_gradient = d_output * (self.input_data > 0) + 0.01 * d_output * (self.input_data <= 0)
        return input_gradient