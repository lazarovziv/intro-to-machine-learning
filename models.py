import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, num_features, classes, epochs=1):
        # num_features includes the extra bias column with the dataset so we'll add it
        self.num_features = num_features
        # values of the multi-class labels
        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_samples = 0
        self.epochs = epochs

        # initializing the weights vectors for every class with values [-1, 1]
        self.weights = np.random.uniform(low=-1, high=1, size=(self.num_classes, self.num_features))
        # weight vectors for pocketing
        self.final_weights = None
        # determined by the size of samples
        self.labels = None

    def init_multi_class_labels(self, y_train, negative_value=-1):
        self.num_samples = y_train.shape[0]
        # initializing the perceptron training labels
        self.labels = np.ones(shape=(self.num_classes, self.num_samples))
        # setting labels' values based on every class in the training data
        for class_idx in range(self.num_classes):
            self.labels[class_idx, np.where(y_train != class_idx)[0]] = negative_value

    def fit(self, X_train, y_train, visualize=True):
        self.init_multi_class_labels(y_train)
        # defining error value for each class for pocket algorithm
        min_errors = np.full(shape=(self.num_classes,), fill_value=np.inf)

        '''
        choosing 16% (arbitrary decision) of the dataset as random samples to calculate errors,
        as we can see from the above cells, the number of occurrences of each digit is 
        distributed almost uniformly, so getting 16% of the samples can distribute the amount of each
        digit in the random sample almost evenly for more general optimization
        '''
        num_random_samples = int(0.16 * self.num_samples)
        # initializing the final_weights here for easier usage with inheritance
        self.final_weights = np.copy(self.weights)
        visualization_iteration_skip = 10
        # error values for every 10th iteration (/sample) for the visualization
        iterations_errors = np.full(shape=(self.num_samples // visualization_iteration_skip, self.num_classes),
                                    fill_value=np.inf)
        # traversing the dataset (per sample approach)
        for epoch in range(self.epochs):
            for t in range(self.num_samples):
                # predicting the current sample with every weight vector using the np.sign method
                # because probability of the dot product to be exactly 0 is very low, and if so, we'll count predictions on the hyperplane
                # to be negative
                y_preds = np.sign(self.weights.dot(X_train[t]))

                # dividing classes based on their predictions
                misclassified_classes_indexes = np.where(y_preds != self.labels[:, t])
                # correcting misclassified weight vectors by multiplying the true sign value by the current sample for each weight vector
                self.weights[misclassified_classes_indexes] += \
                    self.labels[misclassified_classes_indexes, t][0][:, np.newaxis] * X_train[t, :]

                # pocketing every 10 samples
                if t % visualization_iteration_skip == 0:
                    # getting random indexes for the sampling
                    random_samples_indexes = np.random.choice(self.num_samples, num_random_samples)

                    # calculating the errors from the random samples based on the improved vectors
                    y_preds_unsigned = self.weights.dot(X_train[random_samples_indexes, :].T)
                    errors_t = np.sum(
                        np.sign(y_preds_unsigned) != self.labels[:, random_samples_indexes],
                        axis=1) / num_random_samples

                    if visualize:
                        iterations_errors[t // visualization_iteration_skip] = np.copy(errors_t)

                    # pocketing the improved weight vectors and updating the min errors
                    to_improve_classes_indexes = np.where(errors_t < min_errors)[0]

                    self.final_weights[to_improve_classes_indexes] = \
                        np.copy(self.weights[to_improve_classes_indexes])

                    min_errors[to_improve_classes_indexes] = errors_t[to_improve_classes_indexes]

        # visualizing the errors for each class
        if visualize:
            fig, ax = plt.subplots(self.num_classes // 2, 2, figsize=(10, 10))
            fig.tight_layout()
            x_axis = [epoch * visualization_iteration_skip for epoch in range(iterations_errors.shape[0])]
            for i in range(iterations_errors.shape[1]):
                ax[i // 2, i % 2].plot(x_axis, iterations_errors[:, i])
                ax[i // 2, i % 2].set_title(f'Error for class {i}')
            # setting the max value of the y axis to be 0.5 for better readability
            plt.setp(ax, xlim=(0, self.num_samples), ylim=(0, 0.5))
            plt.show()

        print(min_errors)
        # setting the trained weight vectors
        self.weights = np.copy(self.final_weights)

    def predict(self, X_test):
        return np.argmax(X_test @ self.weights.T, axis=1)

    @staticmethod
    def accuracy(self, predictions, y_test):
        return np.sum(predictions == y_test) / len(y_test)


def softmax(s):
    # normalize values to avoid over/underflow of the values (s can't be the zero vector)
    mean = np.mean(s, keepdims=True)
    std = np.std(s, keepdims=True)
    z = (s - mean) / std
    z = np.exp(z)
    return z / np.sum(z, axis=0)


# inheriting from the perceptron class to use the predict and metrics methods
class LogisticRegression(Perceptron):

    def __init__(self, num_features, classes, learning_rate=0.01, epochs=1):
        # initializing variables like the perceptron
        super().__init__(num_features=num_features, classes=classes, epochs=epochs)
        self.lr = learning_rate
        # weights vectors' components are still uniformly distributes between -1 and 1

    def fit(self, X_train, y_train, visualize=True):
        # setting the negative value to be zero for the indicator in the gradient calculation
        super().init_multi_class_labels(y_train=y_train, negative_value=0)

        # need gradients to be of shape (self.num_classes, self.num_features)
        # where gradients[i] = sum((softmax(w_i, x_n) - I[y_n = i]) * x_n)
        for epoch in range(self.epochs):
            # shape = (num_classes x num_samples)
            # row i refers to the exponent of w_i with all the samples
            weights_X_exp = self.weights @ X_train.T

            '''
            1. self.labels is the indicator in the gradient.
            2. multiplying every result (i.e softmax - indicator) by the matching sample
            needed to add a new dimension to multiply each result by the relevant sample for each of the weight vectors
            '''
            gradients = np.sum((softmax(weights_X_exp) - self.labels)[:, :, np.newaxis] * X_train, axis=1)

            # making the weight vectors to go in the opposite direction of the matching gradient self.lr "steps"
            self.weights -= self.lr * gradients

            if visualize:
                pass

    def predict(self, X_test):
        return np.argmax(softmax(X_test @ self.weights.T), axis=1)
