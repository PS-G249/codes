import numpy as np
class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.labels = None

    def forward(self, logits, labels):
        """
        Forward pass for softmax with cross-entropy loss.
        Args:
            logits: Raw scores from the previous layer, shape (batch_size, num_classes)
            labels: One-hot encoded true labels, shape (batch_size, num_classes)
        Returns:
            loss: Cross-entropy loss for the batch
        """
        # Shift logits for numerical stability
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        
        # Compute softmax probabilities
        exp_logits = np.exp(logits_shifted)
        self.softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        #print(self.softmax_output)
        # Store labels for backward pass
        self.labels = labels
        
        # Compute cross-entropy loss
        loss = -np.sum(labels * np.log(self.softmax_output + 1e-9)) / logits.shape[0]
        return loss

    def backward(self):
        """
        Backward pass for softmax with cross-entropy loss.
        Returns:
            gradient: Gradient of the loss with respect to the input logits, shape (batch_size, num_classes)
        """
        # Gradient of cross-entropy loss with respect to logits
        batch_size = self.labels.shape[0]
        gradient = (self.softmax_output - self.labels) / batch_size
        return gradient