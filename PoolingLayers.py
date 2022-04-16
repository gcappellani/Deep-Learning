import numpy as np

import ConvolutionalLayers
from DenseLayers import *
from ActivationFunctions import *
import TensorsOperations as to


class PoolingLayer():
    def __init__(self, in_shape, kernel_shape, stride, operation, activation,
                 padding=None, prev_layer=None, next_layer=None):

        self.prev_layer, self.next_layer = prev_layer, next_layer
        self.in_shape = in_shape if padding is None else [s + 2 * padding for s in in_shape]
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation = np.vectorize(activation)
        self.padding = padding
        self.operation = operation

        self.out_shape = to.get_out_shape(self.in_shape, kernel_shape, stride)
        self.noutputs = self.prev_layer.last_outputs.shape[0]
        self.errors = np.zeros(np.append(self.noutputs, np.array(self.out_shape)))
        self.last_outputs = np.zeros(self.errors.shape)
        self.act_derivatives = np.zeros(self.errors.shape)


    def compute_errors(self):
        ConvolutionalLayers.ConvolutionalLayer.compute_errors(self)
        pass

    def compute_pooling_errors(self):
        pass

    def update_weights(self, alfa, beta, gamma):
        pass


class MinMaxPoolingLayer(PoolingLayer):
    def __init__(self, in_shape, kernel_shape, stride, operation, activation,
                 padding=None, prev_layer=None, next_layer=None):
        super().__init__(in_shape, kernel_shape, stride, None,
                         activation, padding, prev_layer, next_layer)

        if operation == 'max' : self.operation = np.max
        if operation == 'min' : self.operation = np.min
        self.tracking_lists = [[]] * self.noutputs
        self.kernels = np.array([np.zeros(self.kernel_shape)])


    def compute_output(self, tensors=None):
        tensors = self.prev_layer.last_outputs if tensors is None else tensors
        for i in range(len(tensors)):
            if self.padding is not None:
                tensors[i] = np.pad(tensors[i], self.padding, 'constant', constant_values=0)

            output, tracking_list = to.min_max_pooling(tensors[i], self.kernels[0], self.operation, self.stride)

            self.last_outputs[i] = self.activation(output)
            self.act_derivatives[i] = self.activation(output, derivative=True)
            self.tracking_lists[i] = tracking_list

        return self.last_outputs


    def compute_pooling_errors(self):
        for i in range(self.noutputs):
            curr_errors = self.errors[i]
            prev_errors = np.zeros(self.prev_layer.errors[i].shape)
            tracking_list = self.tracking_lists[i]
            for j in range(len(tracking_list)):
                prev_errors[tracking_list[j][0]] += curr_errors[tracking_list[j][1]]
            self.prev_layer.errors[i] = prev_errors


class AveragePoolingLayer(PoolingLayer):
    def __init__(self, in_shape, kernel_shape, stride, activation,
                 padding=None, prev_layer=None, next_layer=None):
        super().__init__(in_shape, kernel_shape, stride, np.average,
                         activation, padding, prev_layer, next_layer)

        self.kernels = np.array([np.ones(self.kernel_shape)])


    def compute_output(self, tensors=None):
        tensors = self.prev_layer.last_outputs if tensors is None else tensors
        for i in range(len(tensors)):
            if self.padding is not None:
                tensors[i] = np.pad(tensors[i], self.padding, 'constant', constant_values=0)

            output = to.avg_pooling(tensors[i], self.kernels[0], self.operation, self.stride)

            self.last_outputs[i] = self.activation(output)
            self.act_derivatives[i] = self.activation(output, derivative=True)

        return self.last_outputs


    def compute_pooling_errors(self):
        for i in range(self.noutputs):

            # Preparing dummy kernel for full convolution
            if len(self.kernel_shape) != len(self.errors[i].shape):
                self.kernel = to.add_axis(self.kernel, len(self.kernel_shape) - len(self.errors[i].shape))

            # Managing stride
            if self.stride != 1:
                self.errors[i] = to.dilate(self.errors[i], self.stride - 1)

            self.prev_layer.errors[i] = to.full_convolution(self.errors[i], self.kernels[0]) / np.prod(self.kernel_shape)