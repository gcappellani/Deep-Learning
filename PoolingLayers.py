import numpy as np
import TensorsOperations as to
from ConvolutionalLayers import ConvolutionalLayer


class PoolingLayer(ConvolutionalLayer):
    def __init__(self, in_shape, kernel_shape, stride, activation, operation,
                 padding=None, prev_layer=None, next_layer=None):

        super().__init__(in_shape, kernel_shape, stride, 1, activation, padding, prev_layer, next_layer)

        self.operation = operation
        self.out_shape = to.get_out_shape(self.in_shape, kernel_shape, stride)

        self.noutputs = self.prev_layer.last_outputs.shape[0]
        self.errors = np.zeros(np.append(self.noutputs, np.array(self.out_shape)))
        self.last_outputs = np.zeros(self.errors.shape)
        self.act_derivatives = np.zeros(self.errors.shape)


    def compute_pooling_errors(self):
        pass

    def update_weights(self, alfa, beta, gamma):
        pass


class MinMaxPoolingLayer(PoolingLayer):
    def __init__(self, in_shape, kernel_shape, stride, activation, operation,
                 padding=None, prev_layer=None, next_layer=None):
        super().__init__(in_shape, kernel_shape, stride, activation,
                         None, padding, prev_layer, next_layer)

        if operation == 'max' : self.operation = np.max
        if operation == 'min' : self.operation = np.min
        self.tracking_lists = [[]] * self.noutputs
        self.kernels = np.array([np.zeros(self.kernel_shape)])


    def compute_output(self, tensors):
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
            self.prev_layer.errors[i] = np.zeros(self.prev_layer.errors[i].shape)
            tracking_list = self.tracking_lists[i]
            for j in range(len(tracking_list)):
                self.prev_layer.errors[tracking_list[j][0]] += self.errors[tracking_list[j][1]]


class AveragePoolingLayer(PoolingLayer):
    def __init__(self, in_shape, kernel_shape, stride, activation,
                 padding=None, prev_layer=None, next_layer=None):
        super().__init__(in_shape, kernel_shape, stride, activation,
                         np.avg, padding, prev_layer, next_layer)

        self.kernels = np.array([np.ones(self.kernel_shape)])


    def compute_output(self, tensors):
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
                to.dilate(self.errors[i], self.stride - 1)

            self.prev_layer.errors[i] = to.full_convolution(self.errors[i], self.kernel) / np.prod(self.kernel_shape)