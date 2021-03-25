import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_prime(x):
    A = sigmoid(x)
    return A * (1 - A)

class NeuralNetwork(object):

    def __init__(self, architecture): #architecture = [4,2,1]
        self.L = architecture.size - 1
        self.n = architecture # n = [4,2,1]

        self.parameters = {}
        self.derivatives = {}

        self.parameters['C'] = 1 #initial cost

        for i in range(1, self.L + 1):

            self.parameters['W'+str(i)] = np.random.randn(self.n[i], self.n[i-1]) * 0.01
            self.parameters['b' + str(i)] = np.ones(self.n[i], 1)
            self.parameters['z' + str(i)] = np.ones(self.n[i] + 1)
            self.parameters['a' + str(i)] = np.ones(self.n[i] + 1)


    def forward_propagate(self, X):

        self.parameters['a0'] = X

        for l in range(1, self.L + 1):
            self.parameters['z' + str(l)] = np.add(np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)]), self.parameters['b' + str(l)])
            self.parameters['a' + str(l)] = sigmoid(self.parameters['z' + str(l)])

    def compute_cost(self, y):
        yhat = self.parameters['a' + str(self.L)]
        self.parameters['C'] = -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))


    def compute_derivatives(self, y):
        #First compute dw, db and dz for the last or 'L' th layer
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str[self.L]] - y

        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))

        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        #Now compute dz, dw and db for each 'l' layer
        for l in range(self.L - 1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(np.transpose(self.parameters['W' + str(l + 1)]), self.derivatives['dz' + str(l + 1)])*sigmoid_prime(self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(self.derivatives['dz' + str(l)], np.transpose(self.parameters['a' + str(l - 1)]))
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]

    def update_parameters(self, alpha):
        for l in range(1, self.L + 1): #we want to include the last layer as well
            self.parameters['W' + str(l)] -= alpha*self.derivatives['dW' + str(l)]
            self.parameters['b' + str(l)] -= alpha*self.derivatives['db' + str(l)]

    def predict(self, X):
        self.forward_propagate(X)
        return self.parameters['a' + str(self.L)] #the aL will now be the predicted value

    def fit(self, X, y, num_iter, alpha=0.01):
        for iter in range(0, num_iter):
            c = 0 #stores cost of the iteration
            n_c = 0 #stores correct predictions

            for i in range(0, X.shape[0]):
              x = X[i].reshape((X[i].size, 1))
              y = Y[i]

              self.forward_propagate(x)
              self.compute_cost(y)
              self.compute_derivatives(y)
              self.update_parameters(alpha)

              c += self.parameters['C']

              y_pred = self.predict(x)

              y_pred = (y_pred > 0.5)

              if(y_pred == y):
                  n_c += 1

            c = c/X.shape[0]

            print('Iteration: ', iter)
            print('Cost: ', c)
            print('Accuracy: ', (n_c/X.shape[0])*100)