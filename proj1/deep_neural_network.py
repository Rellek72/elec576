__author__ = 'thomas_keller'
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
class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_layers, nn_layer_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_layers: the number of layers, not counting input layer
        :param nn_layer_dims: the dimensions of each layer, including in & out
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''

        if nn_layers <= 0:
            raise ValueError("Invalid amount of layers: must be >= 2")
        self.nn_layers = nn_layers

        if any(i <= 0 for i in nn_layer_dims) or len(nn_layer_dims) != nn_layers:
            raise ValueError("Invalid layer size: all must be > 0 and length==nn_layers")
        self.nn_layer_dims = nn_layer_dims

        self.reg_lambda = reg_lambda

        if actFun_type not in ['tanh', 'sigmoid', 'relu']:
            raise ValueError("Invalid activation function. Valid inputs are \"tanh\", \"sigmoid\", or \"relu\"")
        self.actFun_type = actFun_type

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.layers = []
        print("DIMS: ", self.nn_layer_dims)
        # Create and instantiate all layers
        for i in range(self.nn_layers-1):
            self.layers += [Layer(np.random.randn(self.nn_layer_dims[i], self.nn_layer_dims[i+1]) / np.sqrt(self.nn_layer_dims[i]),
                                  np.zeros((1, self.nn_layer_dims[i+1])))]


    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, ReLU, or softmax
        :return: activations
        '''

        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1/(1 + np.exp(-1*z))
        elif type == "relu":
            return np.maximum(0.0, z)
        else:
            return z

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
            return 1

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        for i in range(len(self.layers)):
            self.layers[i].feedforward(X, lambda x: self.actFun(x, self.actFun_type))
            z = self.layers[i].z
            X = self.layers[i].a

        e_zs = np.exp(z - np.expand_dims(z.max(axis=1), axis=1))
        self.denom = np.sum(e_zs, axis=1, keepdims=True)
        self.probs = e_zs / self.denom

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''

        self.feedforward(X)

        log_sm = self.layers[-1].z - np.log(self.denom)
        loss = y * log_sm.T
        data_loss = np.sum(loss)

        # Add regulatization term to loss (optional)
        reg_term = 0.0
        for layer in self.layers[:-1]:
            reg_term += np.sum(np.square(layer.W))
        data_loss += reg_term * (self.reg_lambda / 2)
        return (1. / len(X)) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        print(self.probs)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # Backprop over softmax & output layer
        dz = self.probs - y.T
        dbs = [np.sum(dz, axis=0, keepdims=True)]
        dWs = [np.matmul(self.layers[-2].a.T, dz)]

        # Backprop over remaining layers
        for i in range(self.nn_layers-3, -1, -1):
            W = self.layers[i+1].W
            if (self.nn_layers-3 <= 0):
                a = X
            else:
                a = self.layers[i-1].a
            dz, dW, db = self.layers[i].backprop(dz, W, a, lambda x: self.diff_actFun(x, type=self.actFun_type))
            dWs += [dW]
            dbs += [db]

        return dWs[::-1], dbs[::-1]

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''

        # hot-encode
        y = np.asarray([1-y, y])
        for i in range(0, num_passes):

            # Forward propagation
            self.feedforward(X)
            self.calculate_loss(X, y)

            # Backpropagation
            dWs, dbs = self.backprop(X, y)

            # Add derivatives of regularization terms (biases don't have regularization terms)
            for j in range(len(dWs)):
                dWs[j] += self.reg_lambda * self.layers[j].W

            # Gradient descent parameter update
            for j in range(len(dWs)):
                self.layers[j].W += -epsilon * dWs[j]
                self.layers[j].b += -epsilon * dbs[j]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''

        plot_decision_boundary(lambda x: self.predict(x), X, y)


class Layer(object):
    """
    This class defines and execute the feedforward and backprop operations for a single layer
    """
    def __init__(self, weights, bias):
        '''
        :param weights: the weights scaling the input to this layer
        :param bias: the bias added to the product of the weights and FF input
        '''

        self.W = weights
        self.b = bias
        self.z = None
        self.a = None

    def feedforward(self, X, actFun):
        '''
        feedforward part of the layer
        :param X: input data
        :return:
        '''

        self.z = np.matmul(X, self.W) + self.b
        self.a = actFun(self.z)

        return None

    def backprop(self, dz_pp, w_pp, a_pp, diff_actFun):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param dZpp: the derivative for the output of the previous layer
        :return: dL/dz,dL/dW, dL/db
        '''

        dz = np.matmul(dz_pp, w_pp.T) * diff_actFun(self.z)
        dW = np.matmul(a_pp.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        return dz, dW, db


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()

    dims = [2, 1, 2]
    model = DeepNeuralNetwork(nn_layers=len(dims), nn_layer_dims=dims, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()
