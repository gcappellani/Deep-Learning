import math
import numpy as np


def identity_function(output, derivative=False):
    if derivative :
        return 1.
    return output

def step_function(output):
    return int(output > .0)

def relu(output, derivative=False):
    if derivative :
        return float(output > 0)
    return output if output > 0 else .0

def symmetric_relu(output, derivative=False):
    if derivative :
        return 1 if output > 0 else -1
    return output

def sigmoid(output, derivative=False):
    if derivative :
        return sigmoid(output) * (1 - sigmoid(output))
    return 1.0 / (1.0 + np.exp(-1.0 * output))

def tanh(output, derivative=False):
    if derivative :
        return 1.0 - np.power(np.tanh(output), 2)
    return np.tanh(output)
