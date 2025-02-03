import numpy as np
# Convolutional Layer
class Conv2D:
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        np.random.seed(42)
        self.filters = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(num_filters,)

    def forward(self, input_data):
        self.input_data = np.pad(input_data, ((0, 0), (self.padding, self.padding), 
                                              (self.padding, self.padding), (0, 0)),mode="constant")
        n, h, w, c = self.input_data.shape
        out_height = (h - self.kernel_size) // self.stride + 1
        out_width = (w - self.kernel_size) // self.stride + 1
        self.output = np.zeros((n, out_height, out_width, self.num_filters))
        #print(self.filters)
        for i in range(out_height):
            #print(f"i={i}")
            for j in range(out_width):
                x_slice = self.input_data[:, i * self.stride:i * self.stride + self.kernel_size,
                                          j * self.stride:j * self.stride + self.kernel_size, :]
                for f in range(self.num_filters):
                    #self.output[:, i, j, f] = np.sum(x_slice * self.filters[f], axis=(1, 2, 3)) + self.biases[f]
                    self.output[:, i, j, f] = np.sum(x_slice * self.filters[f].transpose(1, 2, 0), axis=(1, 2, 3)) + self.biases[f]
        k=np.round(self.output, decimals=3)
        return k

    
    def backward(self, d_output, learning_rate):
        batch_size, input_height, input_width, input_channels = self.input_data.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(d_output, axis=(0, 1, 2), keepdims=False)
        d_input = np.zeros_like(self.input_data)

        
        for i in range((input_height - self.kernel_size) // self.stride + 1):
            for j in range((input_width - self.kernel_size) // self.stride + 1):
                x_slice = self.input_data[:, i * self.stride:i * self.stride + self.kernel_size,
                                  j * self.stride:j * self.stride + self.kernel_size, :]
                for f in range(self.num_filters):
                    # Compute gradients for filters
                    d_filters[f] += np.sum(
                        x_slice.transpose(0, 3, 2, 1) * d_output[:, i:i + 1, j:j + 1, f:f + 1],
                        axis=0
                    )
            
                    # Compute gradients for input
                    z = d_output[:, i:i + 1, j:j + 1, f:f + 1].transpose(0, 3, 2, 1)  # Adjusted shape of z

                    # Ensure proper broadcasting
                    d_input[:, i * self.stride:i * self.stride + self.kernel_size,
                            j * self.stride:j * self.stride + self.kernel_size, :] += np.sum(z * self.filters[f])


                # Update filters and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        return d_input

