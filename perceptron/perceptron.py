class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0.0

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
        self.bias += error
        for i, _ in enumerate(self.weights):
            self.weights[i] += error * features[i]

        return error
