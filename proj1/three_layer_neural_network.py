__author__ = 'thomas_keller, tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1/(1 + np.exp(-1*z))
        elif type == "relu":
            return np.maximum(0.0, z)
        else:
            raise ValueError("Invalid activation function. Valid inputs are \"tanh\", \"sigmoid\", or \"relu\"")

        return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''
        if type == 'tanh':
            return 1/np.cosh(2*z)
        elif type == 'sigmoid':
            sig = 1/(1 + np.exp(-1*z))
            return sig*(1-sig)
        elif type == "relu":
            return (z > 0) * 1.0
        else:
            raise ValueError("Invalid activation function. Valid inputs are \"tanh\", \"sigmoid\", or \"relu\"")

        return None

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        X = np.asarray(X)

        self.z1 = np.matmul(X, self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = np.matmul(self.a1, self.W2) + self.b2


        sm = 25

        if sm == 0:
            e_z2 = np.exp(self.z2)
            self.probs = e_z2 / np.sum(e_z2, axis=1, keepdims=True)
        elif sm == 1 or sm == 2:
            if sm == 1:
                self.z_max = np.max(self.z2)
                self.z2_s = self.z2 - self.z_max
            else:
                self.z_max = np.reshape(np.maximum(self.z2.T[0], self.z2.T[1]).T, [len(X), 1])
                self.z2s = self.z2 - self.z_max
            self.e_z2s = np.exp(self.z2s)
            self.e_z2 = np.exp(self.z2)
            self.probs = self.e_z2s / np.sum(self.e_z2s, axis=1, keepdims=True)
        elif sm == 3:
            mid = self.z2 - np.max(self.z2)
            e_z2 = np.exp(mid)
            self.probs = e_z2 / np.sum(e_z2, axis=1, keepdims=True)
        else:
            self.z_max = np.reshape(np.maximum(self.z2.T[0], self.z2.T[1]), [len(X), 1])
            self.z2s = self.z2 - self.z_max
            self.e_z2s = np.exp(self.z2s)
            self.e_z2 = np.exp(self.z2)
            self.z_sum = np.sum(self.e_z2s, axis=1, keepdims=True)
            self.probs = self.e_z2s / self.z_sum
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        N = len(X)

        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))

        mode = 12
        if mode == 0:
            log_sm = np.log(self.probs)
            loss = y * log_sm.T
            data_loss = -1 * np.sum(loss)
        elif mode == 1:
            #if self.printit:
                #print(np.exp(self.z_max))
            log_sm = self.z2s - self.z_max - np.log(np.sum(self.e_z2s, axis=1, keepdims=True))
            #print(log_sm)
            loss = y * log_sm.T
            data_loss = -1 * np.sum(loss)
        elif mode == 2:
            log_sm = self.z2s - np.reshape(np.log(self.z_max / np.sum(self.e_z2, axis=1)), [N, 1])
            loss = y * log_sm.T
        elif mode == 3:
            data_loss = 0.0
            for i in range(N):
                for j in [0, 1]:
                    delta = y[j][i] * np.log(self.probs[i][j])
                    data_loss += delta
            data_loss = -1 * data_loss
        else:
            log_sm = self.z2 - np.log(self.z_sum)
            loss = y * log_sm.T
            data_loss = np.sum(loss)


        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / N) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        N = len(X)
        coef = -1 / N
        coef = 1

        dz2 = self.probs - y.T

        mode = 0
        if (mode == 0):
            db2 = coef * np.sum(dz2, axis=0, keepdims=True)
            dW2 = coef * np.matmul(self.a1.T, dz2)
            dz1 = (np.matmul(dz2, self.W2.T) * self.diff_actFun(self.z1, self.actFun_type))
            db1 = coef * np.sum(dz1, axis=0, keepdims=True)
            dW1 = coef * np.matmul(X.T, dz1)
        elif mode == 1:
            a_prime = self.diff_actFun(self.z1, self.actFun_type)
            db2 = None
            for i in range(N):
                dLdW2 = np.matmul(np.reshape(self.a1[i], [3, 1]), np.reshape(dz2[i], [1, 2]))
                dLdb1 = np.matmul(np.reshape(dz2[i], [1, 2]), self.W2.T) * a_prime[0]
                dLdW1 = np.matmul(np.reshape(X[i], [2, 1]), dLdb1)

                if db2 is None:
                    db2 = np.reshape(dz2[i], [1, 2])
                    dW2 = dLdW2
                    db1 = dLdb1
                    dW1 = dLdW1
                else:
                    db2 += dz2[i]
                    dW2 += dLdW2
                    db1 += dLdb1
                    dW1 += dLdW1

            dW1 = coef * dW1
            dW2 = coef * dW2
            db1 = coef * db1
            db2 = coef * db2

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.

        y = np.asarray(y)
        y_hot = np.asarray([1-y, y])
        self.printit = False
        for i in range(0, num_passes):
            if i == 174:
                self.printit = True
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.calculate_loss(X, y_hot)
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y_hot)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y_hot)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)

def main():
     # generate and visualize Make-Moons dataset
     X, y = generate_data()
     #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
     #plt.show()

     model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='relu')
     model.fit_model(X, y)
     model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()
