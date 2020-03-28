# @Time    : 28/3/20 21:35
# @Author  :  xcTorres
# @FileName: lr.py

"""
    https://github.com/zotroneneis/machine_learning_basics/blob/master/logistic_regression.ipynb
"""
import numpy as np


class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def train(self, X, y_true, n_iters, learning_rate):
        """
        Trains the logistic regression model on given data X and targets y
        :param x:
        :param y_true:
        :param n_iters:
        :param learning_rate:
        :return:
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
            cost = (-1 / n_samples) * np.sum(y_true * np.log(y_predict) + (1-y_true) * np.log(1-y_predict))

            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y_true))
            db = (1 / n_samples) * np.sum(y_predict - y_true)

            self.weights =  self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            costs.append(cost)
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

        return self.weights, self.bias, costs

    def predict(self, X):
        """
            Predicts binary labels for a set of examples X.
        """
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_labels = [1 if elem > 0.5 else 0 for elem in y_predict]

        return np.array(y_predict_labels)[:, np.newaxis]


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def main():
    X, y_true = make_blobs(n_samples=1000, centers=2)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_true)
    plt.title("Dataset")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()

    # Reshape targets to get column vector with shape (n_samples, 1)
    y_true = y_true[:, np.newaxis]
    # Split the data into a training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y_true)

    print(f'Shape X_train: {X_train.shape}')
    print(f'Shape y_train: {y_train.shape}')
    print(f'Shape X_test: {X_test.shape}')
    print(f'Shape y_test: {y_test.shape}')

    regressor = LogisticRegression()
    w_trained, b_trained, costs = regressor.train(X_train, y_train, n_iters=600, learning_rate=0.009)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.arange(600), costs)
    plt.title("Development of cost over training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()

if __name__ == '__main__':
    main()

