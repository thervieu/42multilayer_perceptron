import numpy as np


def initialize_layers(inputs, units, num_layers, classes=2):
    layers = [inputs]
    layers.extend([units for _ in range(num_layers - 2)])  # Hidden layers
    layers.append(classes)
    return layers


class MLP:
    def __init__(self, options):
        self.layers = initialize_layers(options['inputs'], options['units'], options['layers'])
        self.units = options['units']
        self.learning_rate = options['learning_rate']
        self.batch_size = options['batch_size']
        self.epochs = options['epochs']
        self.weights = []
        self.biases = []
        self.activations = []

    def initialize_weights(self):
        # Randomly initialize weights and biases for each layer
        for i in range(1, len(self.layers)):
            input_size = self.layers[i-1]
            output_size = self.layers[i]
            self.weights.append(np.random.randn(output_size, input_size))
            self.biases.append(np.random.randn(output_size, 1))
            self.activations.append('sigmoid')
        self.activations[-1] = 'softmax'


    def set_weights(self, array):
        # Randomly initialize weights and biases for each layer
        self.activations = array[-1]


    def forward_propagation(self, X):
        # Perform forward propagation
        activations = [X.T]
        for i in range(len(self.layers) - 1):
            Z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            A = self.apply_activation(Z, self.activations[i])
            activations.append(A)
        return activations


    def apply_activation(self, Z, activation):
        # Apply the specified activation function
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'softmax':
            return np.exp(Z) / (np.sum(np.exp(Z), axis=0))
        return Z


    def compute_loss(self, y_true, y_pred):
        # Calculate the binary cross-entropy loss with softmax for binary classification
        epsilon = 1e-15
        y_true=y_true
        y_pred = np.clip(y_pred.T, epsilon, 1 - epsilon)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)


    def backward_propagation(self, X, y_true, activations):
        # Perform backward propagation and update weights and biases
        m = X.shape[0]
        dA_prev = 2 * (activations[-1] - y_true.T) / m
        for i in range(len(self.layers) - 2, -1, -1):
            dZ = dA_prev * self.apply_activation_derivative(activations[i + 1], self.activations[i])
            dW = np.dot(dZ, activations[i].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            dA_prev = np.dot(self.weights[i].T, dZ)


    def apply_activation_derivative(self, A, activation):
        # Apply the derivative of the specified activation function
        if activation == 'sigmoid':
            return A * (1 - A)
        elif activation == 'softmax':
            return A * (1 - A)
        return A


    def accuracy(self, y_true, y_pred):
        # Calculate accuracy by comparing predicted classes to ground truth classes
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes)


    def train(self, X, y, validation_data=None):
        self.initialize_weights()
        for epoch in range(1, self.epochs + 1):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                activations = self.forward_propagation(X_batch)
                loss = self.compute_loss(y_batch, activations[-1])

                self.backward_propagation(X_batch, y_batch, activations)

            # Calculate accuracy for training data
            train_predictions = self.predict(X)
            train_accuracy = self.accuracy(y, train_predictions)


            if validation_data:
                X_val, y_val = validation_data
                val_activations = self.forward_propagation(X_val)
                val_loss = self.compute_loss(y_val, val_activations[-1])


                # Calculate accuracy for validation data
                val_predictions = self.predict(X_val)
                val_accuracy = self.accuracy(y_val, val_predictions)
                if epoch%10==0:
                    print(f"Epoch {epoch:03d}/{self.epochs}, loss: {loss:.3f}, val_loss: {val_loss:.3f}, acc: {train_accuracy:.3f}, val_acc: {val_accuracy:.3f}")

            elif epoch%10==0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.3f}, Accuracy: {train_accuracy:.3f}")


    def predict(self, X):
        activations = self.forward_propagation(X)
        return activations[-1].T
