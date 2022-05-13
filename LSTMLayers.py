from ActivationFunctions import *
from DenseLayers import *


class LSTMCell:
    def __init__(self, inputsize, bias):
        self.input_layer = DenseLayer(inputsize, identity_function, bias=bias)
        self.prev_hidden_layer = DenseLayer(inputsize, identity_function, bias=bias)
        self.gates_layer = DenseLayer(inputsize, identity_function, weights=None)

        self.prev_state_layer = DenseLayer(inputsize, identity_function, bias=bias)
        self.output_layer = DenseLayer(inputsize, sigmoid, bias=bias, prev_layer=self.gates_layer)
        self.ignore_layer = DenseLayer(inputsize, sigmoid, bias=bias, prev_layer=self.gates_layer)
        self.forget_layer = DenseLayer(inputsize, sigmoid, bias=bias, prev_layer=self.gates_layer)
        self.select_layer = DenseLayer(inputsize, tanh, bias=bias, prev_layer=self.gates_layer)

        self.state = np.zeros(inputsize)
        self.state_layer = DenseLayer(inputsize, tanh, weights=None)
        self.hidden_layer = DenseLayer(inputsize, identity_function, weights=None)

        self.gates_layer.next_layers = [self.output_layer, self.ignore_layer, self.forget_layer, self.select_layer]


    def compute_output(self):
        self.input_layer.compute_output()
        self.prev_hidden_layer.compute_output()

        self.gates_layer.compute_output(self.input_layer.last_outputs + \
                                        self.prev_hidden_layer.last_outputs)
        gates_outputs = self.gates_layer.last_outputs

        self.output_layer.compute_output(gates_outputs)
        self.ignore_layer.compute_output(gates_outputs)
        self.forget_layer.compute_output(gates_outputs)
        self.select_layer.compute_output(gates_outputs)

        self.prev_state_layer.compute_output(self.state)
        self.state = self.ignore_layer.last_outputs * self.select_layer.last_outputs + \
                     self.forget_layer.last_outputs * self.prev_state_layer.last_outputs
        self.state_layer.compute_output(self.state)

        self.hidden = self.output_layer.last_outputs * self.state_layer.last_outputs
        self.hidden_layer.compute_output(self.hidden)


    def compute_errors(self):
        self.state_layer.compute_errors()
        self.hidden_layer.compute_errors()

        self.state_layer.errors += self.hidden_layer.errors * \
                                   self.output_layer.last_outputs * \
                                   self.state_layer.act_derivatives

        self.output_layer.errors = self.hidden_layer.errors * \
                                   self.state_layer.last_outputs * \
                                   self.output_layer.act_derivatives

        self.ignore_layer.errors = self.state_layer.errors * \
                                   self.select_layer.last_outputs * \
                                   self.ignore_layer.act_derivatives

        self.select_layer.errors = self.state_layer.errors * \
                                   self.ignore_layer.last_outputs * \
                                   self.select_layer.act_derivatives

        self.forget_layer.errors = self.state_layer.errors * \
                                   self.prev_state_layer.last_outputs * \
                                   self.forget_layer.act_derivatives

        self.prev_state_layer.errors = self.state_layer.errors * self.forget_layer.last_outputs

        self.gates_layer.compute_errors()
        self.input_layer.errors = self.gates_layer.errors
        self.prev_hidden_layer.errors = self.gates_layer.errors


    def update_weights(self, alfa, beta, gamma):
        self.output_layer.update_weights(alfa, beta, gamma)
        self.ignore_layer.update_weights(alfa, beta, gamma)
        self.forget_layer.update_weights(alfa, beta, gamma)
        self.select_layer.update_weights(alfa, beta, gamma)
        
        self.prev_state_layer.update_weights(alfa, beta, gamma)
        self.input_layer.update_weights(alfa, beta, gamma)
        self.prev_hidden_layer.update_weights(alfa, beta, gamma)


class LSTMInputLayer:
    def __init__(self, nunits, unit_inputsize):
        self.layers = [DenseInputLayer(unit_inputsize) for _ in range(nunits)]


    def compute_output(self, inputs):
        for layer, input in zip(self.layers, inputs) : layer.compute_output(input)


class LSTMLayer:
    def __init__(self, nunits, unit_inputsize, bias=.0):
        self.nunits = nunits
        self.unit_outputsize = unit_inputsize
        self.units = [LSTMCell(unit_inputsize, bias) for _ in range(nunits)]

        self.units[0].prev_state_layer.prev_layer = self.units[0].state_layer
        self.units[0].prev_hidden_layer.prev_layer = self.units[0].hidden_layer

        for u in range(1, nunits):
            self.units[u].prev_state_layer.prev_layer = self.units[u-1].state_layer
            self.units[u].prev_hidden_layer.prev_layer = self.units[u-1].hidden_layer

        for u in range(nunits-1):
            self.units[u].state_layer.next_layers.append(self.units[u+1].prev_state_layer)
            self.units[u].hidden_layer.next_layers.append(self.units[u+1].prev_hidden_layer)


    def compute_output(self):
        for unit in self.units : unit.compute_output()


    def compute_errors(self):
        for u in range(self.nunits-1, -1, -1) : self.units[u].compute_errors()


    def update_weights(self, alfa, beta, gamma):
        for u in range(self.nunits-1, -1, -1) : self.units[u].update_weights(alfa, beta, gamma)


class LSTMOutputLayer:
    def __init__(self, prev_lstm_layer, activation, outputsize=None,
                 weights='random', randrange=(-.5,.5), bias=.0, next_layers=None):

        self.prev_lstm_layer = prev_lstm_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.inputsize = prev_lstm_layer.unit_outputsize * prev_lstm_layer.nunits
        self.outputsize = self.inputsize if outputsize is None else outputsize
        self.dense_layer = DenseLayer(self.inputsize, activation,
                                      outputsize, weights, randrange, bias)
        self.dense_layer.next_layers = self.next_layers

        self.ewma = self.dense_layer.ewma
        self.weights = self.dense_layer.weights
        self.errors = self.dense_layer.errors
        self.last_outputs = self.dense_layer.last_outputs
        self.act_derivatives = self.dense_layer.act_derivatives


    def compute_output(self, inputs=None):
        if inputs is None:
            inputs = np.array([])
            for unit in self.prev_lstm_layer.units:
                inputs = np.append(inputs, unit.hidden_layer.last_outputs)

        self.dense_layer.compute_output(inputs)
        self.last_outputs = self.dense_layer.last_outputs
        self.act_derivatives = self.dense_layer.act_derivatives

        return self.last_outputs


    def compute_errors(self):
        self.dense_layer.compute_errors()
        self.errors = self.dense_layer.errors


    def update_weights(self, alfa, beta, gamma):
        abs_k = 0
        for unit in self.prev_lstm_layer.units:
            prev_layer = unit.hidden_layer
            for j in range(self.dense_layer.outputsize):
                for rel_k in range(prev_layer.outputsize):
                    weight_update = prev_layer.last_outputs[rel_k] * \
                                    self.dense_layer.act_derivatives[j] * \
                                    self.dense_layer.errors[j]

                    self.dense_layer.weights[j][abs_k + rel_k] += alfa * weight_update + beta * \
                                                                  self.dense_layer.ewma[j][abs_k + rel_k]
                    self.dense_layer.ewma[j][abs_k + rel_k] = gamma * weight_update + (1 - gamma) * \
                                                              self.dense_layer.ewma[j][abs_k + rel_k]
            abs_k += prev_layer.outputsize
        self.weights = self.dense_layer.weights