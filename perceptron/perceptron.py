import random

class Perceptron:
    def __init__(self, num_features, learning_rate=1.0):
        self.num_features = num_features
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_features)] # Initialize the model parameters 
        self.bias = random.uniform(-0.5, 0.5) # with small random numbers instead of 0’s
        self.learning_rate = learning_rate # Use a learning rate for updating the weights and bias unit

    def forward(self, features):
        weighted_sum_z = self.bias
        for i, _ in enumerate(self.weights):
            weighted_sum_z += features[i] * self.weights[i]

        if weighted_sum_z > 0.0:
            prediction = 1
        else:
            prediction = 0

        return prediction

    def update(self, features, true_y):
        prediction = self.forward(features)
        error = true_y - prediction
        self.bias += self.learning_rate * error # Use learning rate for updating bias unit
        for i, _ in enumerate(self.weights):
            self.weights[i] += self.learning_rate * error * features[i] # Use learning rate for updating weights

        return error

    def train(self, x_train, y_train, epochs=10, verbose=False):
        for epoch in range(epochs):
            error_count = 0
            for features, true_y in zip(x_train, y_train):
                error = self.update(features, true_y)
                error_count += abs(error)

            if verbose:
                print(f"Epoch {epoch + 1}, Errors: {error_count}")

            if error_count == 0: # Early-stopping to make the Perceptron more efficient
                break

    def compute_accuracy(self, x_test, y_test):
        correct = 0.0
        for features, true_y in zip(x_test, y_test):
            prediction = self.forward(features)
            correct += int(prediction == true_y)

        return correct / len(y_test)
    
    def decision_boundary(self):
        w1, w2 = self.weights[0], self.weights[1]
        b = self.bias

        x1_min = -20
        x2_min = (-(w1 * x1_min) - b) / w2

        x1_max = 20
        x2_max = (-(w1 * x1_max) - b) / w2

        return x1_min, x1_max, x2_min, x2_max