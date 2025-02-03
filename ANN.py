import numpy as np
import matplotlib.pyplot as plt
from Dense import Dense
from ReLU import ReLU
from Tanh import Tanh
from LeakyReLU import LeakyReLU
from SoftmaxWithCrossEntropy import SoftmaxWithCrossEntropy
class ANN:
    def __init__(self):
        """self.layers = [
                    Dense(1200,600),
                    LeakyReLU(),
                    Dense(600,450),
                    LeakyReLU(),
                    Dense(450,150),
                    LeakyReLU(),
                    Dense(150,4),
        ]"""
        """self.layers = [
            Dense(12288, 1536),
            LeakyReLU(),
            Dense(1536, 768),
            LeakyReLU(),
            Dense(768, 96),
            LeakyReLU(),
            Dense(96, 4),
            #Tanh(),
            #Dense(75,4),
            #Tanh()  # Output size = number of classes
        ]"""
        self.layers = [
            Dense(12288, 768),
            Tanh(),
            Dense(768, 96),
            Tanh(),
            Dense(96, 5),
        ]
        self.loss_function=SoftmaxWithCrossEntropy()

    def forward(self, input_data,y_true):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        loss,pre_class,num=self.loss_function.forward(input_data, y_true)
        return loss,pre_class,num
        #return loss,pre_class

    def backward(self, learning_rate):
        loss_gradient=self.loss_function.backward()
        reversed_layers=list(reversed(self.layers))
        for layer in reversed_layers:
            #print("done")
            loss_gradient = layer.backward(loss_gradient, learning_rate)


    def calc_accuracy(self,pred,pred_test):
        count=0
        for i in range(len(pred)):
            if pred[i]==pred_test[i]:
                count+=1
        return (count/len(pred))*100

    def train(self, x_train, y_train, pred_train,x_test, y_test, pred_test, learning_rate, epochs):
        loss_train=[]
        loss_test=[]
        epoch_no=[]
        acc_train=[]
        acc_test=[]
        for epoch in range(epochs):
            l,predictions,num = self.forward(x_train,y_train)
            loss_train.append(l)
            epoch_no.append(epoch)
            predicted_classes_train = np.argmax(predictions, axis=1)
            #print("Predicted classes:", predicted_classes)
            #loss = self.loss_function.forward( y_train,predictions)
            print(f"Epoch {epoch + 1}, Loss: {l}")
            #loss_gradient = self.loss_function.backward()
            self.backward(learning_rate)
            acc_train.append(self.calc_accuracy(predicted_classes_train,pred_train))

            #testing
            l,predictions,num = self.forward(x_test,y_test)
            loss_test.append(l)
            predicted_classes_test = np.argmax(predictions, axis=1)
            acc_test.append(self.calc_accuracy(predicted_classes_test,pred_test))
        #return loss,epoch_no
        return loss_train, predicted_classes_train,epoch_no,acc_train,loss_test, predicted_classes_test,epoch_no,acc_test
        