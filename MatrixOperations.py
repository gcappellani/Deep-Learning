import numpy as np
from Exceptions import ShapeException


def get_out_shape(input_shape, kernel_shape, stride):
    out_shape = []
    for i in range(len(input_shape)):
        size = ((input_shape[i] - kernel_shape[i]) / stride) + 1
        if float.is_integer(size) is False:
            raise ShapeException(input_shape, kernel_shape, stride)
        out_shape.append(int(size))
    return np.array(out_shape)


def operate(i, temp, in_tensor, kernel, out_tensor, stride, bias):
    for j in range(0, in_tensor.shape[i]-kernel.shape[i]+1, stride):
        if i == len(out_tensor.shape) - 1:
            for k in range(0, in_tensor.shape[-1]-kernel.shape[i]+1, stride):
                temp[-1] = k
                out_index = tuple(int(index/stride) for index in temp)  # Beautify
                in_index = tuple(slice(a, b) for a, b in zip(temp, temp+kernel.shape))
                out_tensor[out_index] = np.sum(in_tensor[in_index] * kernel) + bias
            return
        else:
            operate(i + 1, temp, in_tensor, kernel, out_tensor, stride, bias)
            temp[i] += stride
    temp[i] = 0


def operate_recursive(i, temp, in_tensor, kernel, out_tensor, stride, bias):
    for j in range(0, in_tensor.shape[i]-kernel.shape[i]+1, stride):
        if i == len(out_tensor.shape):
            return
        if i == len(out_tensor.shape)-1 :
            temp[-1] = j
            out_index = tuple(int(index / stride) for index in temp)  # Beautify
            in_index = tuple(slice(a, b) for a, b in zip(temp, temp + kernel.shape))
            out_tensor[out_index] = np.sum(in_tensor[in_index] * kernel) + bias
        operate(i + 1, temp, in_tensor, kernel, out_tensor, stride, bias)
        temp[i] += stride
    temp[i] = 0


def convolution(in_tensor, kernel, stride=1, bias=0):
    out_shape = get_out_shape(in_tensor.shape, kernel.shape, stride)
    out_tensor = np.zeros(out_shape)

    temp = np.array([0] * len(out_tensor.shape))
    operate_recursive(0, temp, in_tensor, kernel, out_tensor, stride, bias)

    return out_tensor


def full_convolution(in_tensor, kernel, stride=1, bias=0):
    out_shape = in_tensor.shape
    out_tensor = np.zeros(out_shape)

    pad_width = [(s-1, s-1) for s in kernel.shape]
    in_tensor = np.pad(in_tensor, pad_width, 'constant', constant_values=0)

    temp = np.array([0] * len(out_tensor.shape))
    operate_recursive(0, temp, in_tensor, kernel, out_tensor, stride, bias)

    return out_tensor

