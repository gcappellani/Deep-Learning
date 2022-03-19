import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from TensorsOperations import *


if __name__ == '__main__':

    #(train_images, train_labels), (test_images, test_labels) = tfds.cifar10.load_data()

    in_shape = [3,3]
    kernel_shape = [2,2,2]

    tensor = np.arange(9).reshape(in_shape)

    new_in_shape = [1 for _ in range(len(kernel_shape) - len(in_shape))]
    for s in in_shape : new_in_shape.append(s)
    tensor = tensor.reshape(new_in_shape)

    pad_width = [(s - 1, s - 1) for s in kernel_shape]
    tensor = np.pad(tensor, pad_width, 'constant', constant_values=0)


    out_shape = np.array([3,3])

    print(out_shape.shape)









