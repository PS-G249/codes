import numpy as np
# Flatten Layer
class Flatten:
    def forward(self, input_data):
        self.input_data_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, d_output,learning_rate):
        return d_output.reshape(self.input_data_shape)