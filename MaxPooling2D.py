import numpy as np
# MaxPooling Layer
class MaxPooling2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input_data = input_data
        n, h, w, c = input_data.shape
        out_height = (h - self.pool_size) // self.stride + 1
        out_width = (w - self.pool_size) // self.stride + 1
        self.output = np.zeros((n, out_height, out_width, c))

        for i in range(out_height):
            for j in range(out_width):
                x_slice = input_data[:, i * self.stride:i * self.stride + self.pool_size,
                                     j * self.stride:j * self.stride + self.pool_size, :]
                self.output[:, i, j, :] = np.max(x_slice, axis=(1, 2))
        #print('max pooling output shape',self.output.shape)
        return self.output

    def backward(self, d_output,learning_rate):
        batch_size, channels, out_height, out_width = d_output.shape
        #print("MAx back input pooling shape: ",d_output.shape)
        d_input = np.zeros_like(self.input_data)

        pool_height, pool_width = self.pool_size,self.pool_size

        #print('batch size',batch_size)
        for b in range(batch_size):
            #print("b dims",b)
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = max(h * self.stride, 0)
                        h_end = min(h_start + pool_height, self.input_data.shape[2])
                        w_start = max(w * self.stride, 0)
                        w_end = min(w_start + pool_width, self.input_data.shape[3])

                        # Region of the input corresponding to this output
                        region = self.input_data[b, c, h_start:h_end, w_start:w_end]
                        if region.size > 0:
                            max_value = np.max(region)

                            # Distribute gradient to the maximum value position
                            for i in range(min(pool_height, region.shape[0])):
                                for j in range(min(pool_width, region.shape[1])):
                                    if region[i, j] == max_value:
                                        d_input[b, c, h_start + i, w_start + j] += d_output[b, c, h, w]

        #print('max pooling back output',d_input.shape)
        return d_input