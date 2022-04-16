import numpy as np


class DenseInputLayer:
    def __init__(self, inputsize, next_layer=None):
        self.next_layer = next_layer
        self.outputsize = inputsize

    def compute_output(self, inputs):
        self.last_outputs = inputs
        return inputs


class FlattenLayer:
    def __init__(self, input_shape, prev_layer=None, next_layer=None):
        self.prev_layer, self.next_layer = prev_layer, next_layer
        self.input_shape = input_shape
        self.outputsize = np.prod(input_shape)


    def compute_output(self):
        self.last_outputs = self.prev_layer.last_outputs.reshape(self.outputsize)
        self.act_derivatives = self.prev_layer.act_derivatives
        return self.last_outputs


    def compute_errors(self):
        self.errors = np.zeros(self.outputsize)
        for j in range(self.outputsize):
            perceptron_error = .0
            for k in range(self.next_layer.outputsize):
                perceptron_error += self.next_layer.errors[k] * \
                                    self.next_layer.act_derivatives[k] * \
                                    self.next_layer.weights[k][j]
            self.errors[j] = perceptron_error

        self.errors = self.errors.reshape(self.input_shape)


    def update_weights(self, alfa, beta, gamma):
        pass


class DenseLayer:
    def __init__(self, inputsize, activation,
                 outputsize=None, weights='random', randrange=(-.5,.5), bias=.0, prev_layer=None, next_layers=None):

        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.inputsize = inputsize
        self.outputsize = inputsize if outputsize is None else outputsize
        self.activation = np.vectorize(activation)
        self.bias = bias

        self.ewma = np.zeros((self.outputsize, self.inputsize))
        self.errors = np.zeros(self.outputsize)
        self.last_outputs = np.zeros(self.outputsize)
        self.act_derivatives = np.zeros(self.outputsize)

        if weights == 'random' :
            self.weights = np.random.uniform(randrange[0], randrange[1], (self.outputsize, self.inputsize))
        elif weights is None :
            self.weights = None


    def compute_output(self, inputs=None):
        inputs = self.prev_layer.last_outputs if inputs is None else inputs
        output = inputs if self.weights is None else np.dot(self.weights, inputs) + self.bias

        self.last_outputs = self.activation(output)
        self.act_derivatives = self.activation(output, derivative=True)

        return self.last_outputs


    def compute_errors(self):
        self.errors.fill(0)
        for next_layer in self.next_layers:
            for j in range(self.outputsize):
                perceptron_error = .0
                for k in range(next_layer.outputsize):
                    perceptron_error += next_layer.errors[k] * \
                                        next_layer.act_derivatives[k] * \
                                        next_layer.weights[k][j]
                self.errors[j] += perceptron_error


    def update_weights(self, alfa, beta, gamma):
        for j in range(self.outputsize):
            for k in range(self.prev_layer.outputsize):
                weight_update = self.prev_layer.last_outputs[k] * \
                                self.act_derivatives[j] * \
                                self.errors[j]

                self.weights[j][k] += alfa * weight_update + beta * self.ewma[j][k]
                self.ewma[j][k] = gamma * weight_update + (1 - gamma) * self.ewma[j][k]