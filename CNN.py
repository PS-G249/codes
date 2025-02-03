import numpy as np
from Conv2D import Conv2D
from ReLU import ReLU
from LeakyReLU import LeakyReLU
from Tanh import Tanh
from MaxPooling2D import MaxPooling2D
from Flatten import Flatten
from Dense import Dense
from SoftmaxWithCrossEntropy import SoftmaxWithCrossEntropy
# Assembling the CNN
class CNN:
    def __init__(self):
        '''self.layers = [
            Conv2D(input_channels=3, num_filters=8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_dim=8 * 64 * 64, output_dim=4),  # Adjust based on input dimensions
        ]'''
        '''self.layers = [
            Conv2D(input_channels=3, num_filters=8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),
            Conv2D(input_channels=8, num_filters=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_dim=16 * 32 * 32, output_dim=64),
            ReLU(),
            Dense(input_dim=64, output_dim=4),  # 4 output classes
        ]'''

        '''self.layers = [
            Conv2D(input_channels=3, num_filters=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),
            Conv2D(input_channels=32, num_filters=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_dim=16 * 32 * 32, output_dim=64),
            ReLU(),
            Dense(input_dim=64, output_dim=4),  # 4 output classes
        ]'''

        self.layers = [
            Conv2D(input_channels=3, num_filters=32, kernel_size=3, stride=1, padding=1),
            Tanh(),
            MaxPooling2D(pool_size=2, stride=2),
            Conv2D(input_channels=32, num_filters=64, kernel_size=3, stride=1, padding=1),
            Tanh(),
            MaxPooling2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_dim=64 * 16 * 16, output_dim=64),
            #Dense(input_dim=64 * 32 * 32, output_dim=64),
            Tanh(),
            Dense(input_dim=64, output_dim=5),  # 4 output classes
        ]
        self.loss_function=SoftmaxWithCrossEntropy()

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, loss_gradient, learning_rate):
        reversed_layers=list(reversed(self.layers))
        for layer in reversed_layers:
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def accuracy(self, y_train,y_pred):
        count=0
        for i in range(len(y_train)):
            if y_train[i]==y_pred[i]:
                count+=1
        acc=(count/len(y_train))*100
        return acc

    '''def train(self, x_train, y_train,pred, learning_rate, epochs):
        predicted_classes=[]
        loss_arr=[]
        acc=[]
        for epoch in range(epochs):
            predictions = self.forward(x_train)
            predicted_classes = np.argmax(predictions, axis=1)
            acc.append(self.accuracy(pred,predicted_classes))
            print("Predicted classes:", predicted_classes)
            loss = self.loss_function.forward(predictions, y_train)
            loss_arr.append(loss)
            print(f"Epoch {epoch + 1}, Loss: {loss}")
            loss_gradient = self.loss_function.backward()
            self.backward(loss_gradient, learning_rate)
        return predicted_classes,loss_arr,epochs,acc'''
    
    def train(self, x_train, y_train, pred_train,x_test, y_test, pred_test, learning_rate, epochs):
        loss_train=[]
        loss_test=[]
        acc_train=[]
        acc_test=[]
        for epoch in range(epochs):
            predictions = self.forward(x_train)
            l=self.loss_function.forward(predictions,y_train)
            loss_train.append(l)
            print((np.array(loss_train)).size)
            predicted_classes_train = np.argmax(predictions, axis=1)
            acc_train.append(self.accuracy(pred_train,predicted_classes_train))
            #print("Predicted classes:", predicted_classes)
            #loss = self.loss_function.forward( y_train,predictions)
            print(f"Epoch {epoch + 1}, Loss: {l}")
            #loss_gradient = self.loss_function.backward()
            loss_gradient = self.loss_function.backward()
            self.backward(loss_gradient, learning_rate)

            #testing
            predictions = self.forward(x_test)
            l=self.loss_function.forward(predictions,y_test)
            loss_test.append(l)
            predicted_classes_test = np.argmax(predictions, axis=1)
            acc_test.append(self.accuracy(predicted_classes_test,pred_test))
        #return loss,epoch_no
        return loss_train, predicted_classes_train,acc_train,loss_test, predicted_classes_test,acc_test