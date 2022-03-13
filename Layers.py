import numpy as np


class InputLayer:
    def __init__(self, inputsize):
        self.size = inputsize

    def compute_output(self, inputs):
        self.last_outputs = inputs
        return inputs


class DenseLayer:
    def __init__(self, inputsize, activation, size=None, weights=None, randrange=None, bias=.0):
        self.inputsize = inputsize
        self.size = inputsize if size is None else size
        self.activation = activation
        self.bias = bias

        self.last_outputs = np.zeros(self.size)
        self.act_derivatives = np.zeros(self.size)
        self.errors = np.zeros(self.size)
        self.ewma = np.zeros((self.size, self.inputsize))

        if weights == 'random' :
            self.weights = np.random.uniform(randrange[0], randrange[1], (self.size, self.inputsize))
        elif weights is None :
            self.weights = None


    def compute_output(self, inputs):
        output = np.dot(self.weights, inputs) + self.bias
        self.last_outputs = self.activation(output)
        self.act_derivatives = self.activation(output, derivative=True)

        return self.last_outputs


    def compute_errors(self, next_layer):
        for j in range(self.size):
            perceptron_error = .0
            for k in range(next_layer.size):
                perceptron_error += next_layer.errors[k] * \
                                    next_layer.act_derivatives[k] * \
                                    next_layer.weights[k][j]
            self.errors[j] = perceptron_error


    def update_weights(self, alfa, beta, gamma, prev_layer):
        for j in range(self.size):
            for k in range(prev_layer.size):
                weight_update = prev_layer.last_outputs[k] * \
                                self.act_derivatives[j] * \
                                self.errors[j]

                self.weights[j][k] += alfa * weight_update + beta * self.ewma[j][k]

                self.ewma[j][k] = gamma * weight_update + (1 - gamma) * self.ewma[j][k]


class ConvolutionalLayer():
    def __init__(self, in_shape, kernel_shape, stride, nkernel, padding=None):
        self.in_shape = in_shape if padding is None else [s + 2*padding for s in in_shape]
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.nkernel = nkernel
        self.padding = padding

        self.out_shape = self.get_out_shape()




    def compute_output(self, tensor):
        inputs = np.pad(tensor, self.padding, 'constant', constant_values=0)
