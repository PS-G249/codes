import numpy as np
class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.labels = None

    def forward(self, logits, labels):
        # Shift logits for numerical stability
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        
        # Compute softmax probabilities
        exp_logits = np.exp(logits_shifted)
        self.softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predicted_classes = np.argmax(self.softmax_output, axis=1)
        #print("Predicted classes:", predicted_classes)
        #print(self.softmax_output)
        # Store labels for backward pass
        self.labels = labels
        
        # Compute cross-entropy loss
        loss = -np.sum(labels * np.log(self.softmax_output + 1e-9)) / logits.shape[0]
        #print(predicted_classes.shape)
        #return loss,predicted_classes[0]
        #return loss,self.softmax_output,predicted_classes[0]
        return loss,self.softmax_output,predicted_classes

    def backward(self):
        # Gradient of cross-entropy loss with respect to logits
        batch_size = self.labels.shape[0]
        gradient = (self.softmax_output - self.labels) / batch_size
        return gradient