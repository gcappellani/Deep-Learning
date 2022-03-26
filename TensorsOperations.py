import numpy as np
from Exceptions import ShapeException


def dilate(tensor, padding, value=0):
    for i in range(len(tensor.shape)) :
        for j in range(padding) :
            tensor = np.insert(tensor, range(1, tensor.shape[i], j+1), value, axis=i)

    return tensor


def get_out_shape(input_shape, kernel_shape, stride):
    out_shape = []
    for i in range(len(input_shape)):
        size = ((input_shape[i] - kernel_shape[i]) / stride) + 1
        if float.is_integer(size) is False:
            raise ShapeException(input_shape, kernel_shape, stride)
        out_shape.append(int(size))

    return np.array(out_shape)


def add_axis(tensor, naxis):
    new_shape = [1 for _ in range(naxis)]
    for s in tensor.shape: new_shape.append(s)
    return tensor.reshape(new_shape)


def operate(i, temp, in_tensor, kernel, out_tensor, stride, bias):
    for j in range(0, in_tensor.shape[i]-kernel.shape[i]+1, stride):
        if i == len(out_tensor.shape)-1 :
            temp[-1] = j
            out_index = tuple(int(index / stride) for index in temp)
            in_index = tuple(slice(a, b) for a, b in zip(temp, temp + kernel.shape))
            out_tensor[out_index] = np.sum(in_tensor[in_index] * kernel) + bias
        elif i < len(out_tensor.shape)-1 :
            operate(i + 1, temp, in_tensor, kernel, out_tensor, stride, bias)
            temp[i] += stride
    temp[i] = 0


def convolution(in_tensor, kernel, stride=1, bias=0):
    out_shape = get_out_shape(in_tensor.shape, kernel.shape, stride)
    out_tensor = np.zeros(out_shape)

    temp = np.array([0] * len(out_tensor.shape))
    operate(0, temp, in_tensor, kernel, out_tensor, stride, bias)

    return out_tensor


def full_convolution(in_tensor, kernel, stride=1, bias=0):

    pad_width = [(s-1, s-1) for s in kernel.shape]
    in_tensor = np.pad(in_tensor, pad_width, 'constant', constant_values=0)

    out_shape = in_tensor.shape
    out_tensor = np.zeros(out_shape)

    temp = np.array([0] * len(out_tensor.shape))
    operate(0, temp, in_tensor, kernel, out_tensor, stride, bias)

    return out_tensor


def operate_and_get_mask(i, mask, temp, in_tensor, kernel, out_tensor, stride, operation):
    for j in range(0, in_tensor.shape[i]-kernel.shape[i]+1, stride):
        if i == len(out_tensor.shape)-1 :
            temp[-1] = j
            out_index = tuple(int(index / stride) for index in temp)
            in_index = tuple(slice(a, b) for a, b in zip(temp, temp + kernel.shape))

            target = operation(in_tensor[in_index])
            out_tensor[out_index] = target
            if mask is not None :
                target_index = np.where(in_tensor[in_index] == target)
                mask[tuple(a + b for a,b in zip(temp, target_index))] = 1

        elif i < len(out_tensor.shape)-1 :
            operate_and_get_mask(i + 1, mask, temp, in_tensor, kernel, out_tensor, stride, operation)
            temp[i] += stride
    temp[i] = 0


def pooling(in_tensor, kernel, operation, stride=1):
    mask = np.zeros(in_tensor.shape)
    if operation == "max" :
        operation = np.max
    elif operation == "min" :
        operation = np.min
    elif operation == "avg" :
        operation = np.average
        mask = None

    out_shape = get_out_shape(in_tensor.shape, kernel.shape, stride)
    out_tensor = np.zeros(out_shape)

    temp = np.array([0] * len(out_tensor.shape))
    operate_and_get_mask(0, mask, temp, in_tensor, kernel, out_tensor, stride, operation)

    return out_tensor, mask


