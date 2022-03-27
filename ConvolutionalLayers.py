import numpy as np
import TensorsOperations as to
from PoolingLayers import PoolingLayer


class ConvolutionalInputLayer:
    def __init__(self, inputsize, next_layer=None):
        self.next_layer = next_layer
        self.size = inputsize

    def compute_output(self, inputs):
        inputs = to.add_axis(inputs, 1)
        self.last_outputs = inputs
        return inputs


class ConvolutionalLayer():
    def __init__(self, in_shape, kernel_shape, stride, nkernel, activation,
                 bias=0, randrange=None, padding=None, prev_layer=None, next_layer=None):

        self.prev_layer, self.next_layer = prev_layer, next_layer
        self.in_shape = in_shape if padding is None else [s + 2*padding for s in in_shape]
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.nkernel = nkernel
        self.activation = activation
        self.bias = bias
        self.randrange = randrange if randrange is not None else (-.5,.5)
        self.padding = padding

        self.out_shape = to.get_out_shape(self.in_shape, kernel_shape, stride)
        self.kernels = np.array([np.random.uniform(randrange[0], randrange[1], kernel_shape) for _ in nkernel])
        self.ewma = np.zeros(self.kernels.shape)

        self.noutputs = self.prev_layer.last_outputs.shape[0] * self.nkernel
        self.errors = np.zeros(np.append(self.noutputs, np.array(self.out_shape)))
        self.last_outputs = np.zeros(self.errors.shape)
        self.act_derivatives = np.zeros(self.errors.shape)


    def compute_output(self, tensors):
        for i in range(len(tensors)) :
            if self.padding is not None :
                tensors[i] = np.pad(tensors[i], self.padding, 'constant', constant_values=0)
            for j in range(self.nkernel) :
                output = to.convolution(tensors[i], self.kernels[j], self.stride, self.bias)
                self.last_outputs[i*self.nkernel + j] = self.activation(output)
                self.act_derivatives[i*self.nkernel + j] = self.activation(output, derivative=True)

        return self.last_outputs


    def compute_errors(self):
        if isinstance(self.next_layer, PoolingLayer):
            self.next_layer.compute_pooling_errors()
            return
        for i in range(len(self.errors)) :
            self.errors[i] = np.zeros(self.errors[i].shape)
            for j in range(self.next_layer.nkernel) :
                # Flipping kernel
                flipped_kernel = self.next_layer.kernels[j]
                for axis in range(len(self.next_layer.kernel_shape)):
                    flipped_kernel = np.flip(flipped_kernel, axis)

                loss_grad = self.next_layer.errors[i*self.next_layer.nkernel + j] \
                                  * self.next_layer.act_derivatives[i*self.next_layer.nkernel + j]

                # Preparing flipped kernel for full convolution
                if len(flipped_kernel.shape) != len(loss_grad.shape) :
                    flipped_kernel = to.add_axis(flipped_kernel, len(flipped_kernel.shape) != len(loss_grad.shape))

                # Managing stride
                if self.next_layer.stride != 1 :
                    to.dilate(loss_grad, self.next_layer.stride - 1)

                self.errors[i] += to.full_convolution(loss_grad, flipped_kernel)


    def update_weights(self, alfa, beta, gamma):
        for i in range(self.nkernel):
            kernel_update = np.zeros(self.kernel_shape)
            for j in range(len(self.prev_layer.last_outputs)):
                loss_grad = self.errors[j*self.nkernel + i] \
                            * self.act_derivatives[j*self.nkernel + i]

                # Preparing loss gradients for convolution
                if len(loss_grad.shape) != len(self.prev_layer.last_outputs[j].shape):
                    loss_grad = to.add_axis(loss_grad,
                                    len(self.prev_layer.last_outputs[j].shape) - len(loss_grad.shape))

                # Managing stride
                if self.stride != 1:
                    to.dilate(loss_grad, self.stride - 1)

                kernel_update += to.convolution(self.prev_layer.last_outputs[j], loss_grad)

            self.kernels[i] += alfa * kernel_update + beta * self.ewma[i]
            self.ewma[i] = gamma * kernel_update + (1 - gamma) * self.ewma[i]