import numpy as np
import TensorsOperations as to


class DenseInputLayer:
    def __init__(self, inputsize):
        self.size = inputsize

    def compute_output(self, inputs):
        self.last_outputs = inputs
        return inputs


class ConvolutionalInputLayer:
    def __init__(self, inputsize):
        self.size = inputsize

    def compute_output(self, inputs):
        inputs = to.add_axis(inputs, 1)
        self.last_outputs = inputs
        return inputs


class FlatLayer:
    def __init__(self, input_shape, outputsize):
        self.input_shape = input_shape
        self.outputsize = outputsize


    def compute_output(self, inputs):
        return inputs.reshape(self.outputsize)


    def compute_errors(self, next_layer):
        self.errors = next_layer.errors.reshape(self.input_shape)


class DenseLayer:
    def __init__(self, inputsize, activation, size=None, weights=None, randrange=None, bias=.0):
        self.inputsize = inputsize
        self.size = inputsize if size is None else size
        self.activation = activation
        self.bias = bias

        self.last_outputs = np.zeros(self.size)       # to be removed?
        self.act_derivatives = np.zeros(self.size)    # same?
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
    def __init__(self, in_shape, kernel_shape, stride, nkernel, activation,
                 bias=0, randrange=None, padding=None):

        self.in_shape = in_shape if padding is None else [s + 2*padding for s in in_shape]
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.nkernel = nkernel
        self.activation = activation
        self.bias = bias
        self.randrange = randrange if randrange is not None else (-.5,.5)
        self.padding = padding

        self.out_shape = to.get_out_shape(in_shape, kernel_shape, stride)
        self.kernels = np.array([np.random.uniform(randrange[0], randrange[1], kernel_shape)])
        self.ewma = np.copy(self.kernels)
        self.errors = np.zeros(in_shape)
        self.last_outputs = np.zeros(np.append(len(self.in_shape), np.array(self.out_shape)))
        self.act_derivatives = np.zeros(self.last_outputs.shape)


    def compute_output(self, tensors):
        for i in range(len(tensors)) :
            if self.padding is not None :
                tensors[i] = np.pad(tensors[i], self.padding, 'constant', constant_values=0)
            for j in range(self.nkernel) :
                output = to.convolution(tensors[i], self.kernels[j], self.stride, self.bias)
                self.last_outputs[i*self.nkernel + j] = self.activation(output)
                self.act_derivatives[i*self.nkernel + j] = self.activation(output, derivative=True)

        return self.last_outputs


    def compute_errors(self, next_layer):
        for i in range(len(self.errors)) :
            self.errors[i] = np.zeros(self.errors[i].shape)
            for j in range(next_layer.nkernel) :
                # Flipping kernel
                flipped_kernel = next_layer.kernels[j]
                for axis in range(len(next_layer.kernel_shape)):
                    flipped_kernel = np.flip(flipped_kernel, axis)

                loss_grad = next_layer.errors[i*next_layer.nkernel + j] \
                                  * next_layer.act_derivatives[i*next_layer.nkernel + j]

                # Preparing loss gradients for full convolution
                if len(loss_grad.shape) != len(next_layer.kernel_shape) :
                    loss_grad = to.add_axis(loss_grad, len(next_layer.kernel_shape) - len(loss_grad.shape))

                # Managing stride
                if next_layer.stride != 1 :
                    to.dilate(loss_grad, next_layer.stride - 1)

                self.errors[i] += to.full_convolution(loss_grad, flipped_kernel)


    def update_weights(self, alfa, beta, gamma, prev_layer):
        for i in range(self.nkernel):
            kernel_update = .0
            for j in range(len(prev_layer.last_outputs)):
                loss_grad = self.errors[j*self.nkernel + i] \
                            * self.act_derivatives[j*self.nkernel + i]

                # Preparing loss gradients for convolution
                if len(loss_grad.shape) != len(prev_layer.last_outputs[j].shape):
                    loss_grad = to.add_axis(loss_grad,
                                    len(prev_layer.last_outputs[j].shape) - len(loss_grad.shape))

                # Managing stride
                if self.stride != 1:
                    to.dilate(loss_grad, self.stride - 1)

                kernel_update += to.convolution(prev_layer.last_outputs[j], loss_grad)

            self.kernels[i] += alfa * kernel_update + beta * self.ewma[i]
            self.ewma[i] = gamma * kernel_update + (1 - gamma) * self.ewma[i]
