import numpy as np
from Layer import Layer
class Dense(Layer):
    def __init__(self,input_size,output_size):
        np.random.seed(42)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)  # He initialization
        #self.weights=np.random.randn(input_size,output_size)*0.1
        self.bias=np.random.randn(1,output_size)
    def forward(self,input_ar):
        self.input_ar=input_ar
        #l=np.dot(self.input_ar,self.weights)+self.bias
        #print(l.shape)
        return np.dot(self.input_ar,self.weights)+self.bias
    def backward(self,output_gradient,learning_rate):
        dw=np.dot(self.input_ar.T,output_gradient)
        input_gradient=np.dot(output_gradient,self.weights.T)
        self.weights -= learning_rate * dw
        db = np.sum(output_gradient, axis=0, keepdims=True)
        self.bias-=learning_rate*db
        return input_gradient