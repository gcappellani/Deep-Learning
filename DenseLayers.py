import numpy as np


class DenseInputLayer:
    def __init__(self, inputsize, next_layer=None):
        self.next_layer = next_layer
        self.size = inputsize

    def compute_output(self, inputs):
        self.last_outputs = inputs
        return inputs


class FlattenLayer:
    def __init__(self, input_shape, outputsize, prev_layer=None, next_layer=None):
        self.prev_layer, self.next_layer = prev_layer, next_layer
        self.input_shape = input_shape
        self.outputsize = outputsize


    def compute_output(self, inputs):
        return inputs.reshape(self.outputsize)

    def compute_errors(self):
        self.errors = self.next_layer.errors.reshape(self.input_shape)
        self.act_derivatives = self.next_layer.act_derivatives.reshape(self.input_shape)

    def update_weights(self):
        self.last_outputs = self.prev_layer.last_outputs.reshape(self.outputsize)


class DenseLayer:
    def __init__(self, inputsize, activation,
                 size=None, weights=None, randrange=None, bias=.0, prev_layer=None, next_layer=None):

        self.prev_layer, self.next_layer = prev_layer, next_layer
        self.inputsize = inputsize
        self.size = inputsize if size is None else size
        self.activation = activation
        self.bias = bias

        self.ewma = np.zeros((self.size, self.inputsize))
        self.errors = np.zeros(self.size)
        self.last_outputs = np.zeros(self.size)
        self.act_derivatives = np.zeros(self.size)

        if weights == 'random' :
            self.weights = np.random.uniform(randrange[0], randrange[1], (self.size, self.inputsize))
        elif weights is None :
            self.weights = None


    def compute_output(self, inputs):
        output = np.dot(self.weights, inputs) + self.bias
        self.last_outputs = self.activation(output)
        self.act_derivatives = self.activation(output, derivative=True)

        return self.last_outputs


    def compute_errors(self):
        for j in range(self.size):
            perceptron_error = .0
            for k in range(self.next_layer.size):
                perceptron_error += self.next_layer.errors[k] * \
                                    self.next_layer.act_derivatives[k] * \
                                    self.next_layer.weights[k][j]
            self.errors[j] = perceptron_error


    def update_weights(self, alfa, beta, gamma):
        for j in range(self.size):
            for k in range(self.prev_layer.size):
                weight_update = self.prev_layer.last_outputs[k] * \
                                self.act_derivatives[j] * \
                                self.errors[j]

                self.weights[j][k] += alfa * weight_update + beta * self.ewma[j][k]

                self.ewma[j][k] = gamma * weight_update + (1 - gamma) * self.ewma[j][k]
