from DenseLayers import *
from ConvolutionalLayers import *
from LSTMLayers import *


class NeuralNetwork:
    def __init__(self):
        self.depth = 0
        self.layers = []


    # For no momentum set beta = 0
    # For simple momentum set beta != 0 and gamma = 1
    # For momentum with exponentially weighted moving average
    # set beta != 0 and gamma != 1
    def train(self, X, Y, epochs, alfa=.1, beta=.3, gamma=.7):
        for epoch in range(epochs) :
            i = 1
            for inputs, outputs in zip(X, Y) :
                self.train_on_item(inputs, outputs, alfa, beta, gamma)
                print("trained item " + str(i))
                i += 1
            print("epoch " + str(epoch+1) + " done")


    def train_on_item(self, inputs, outputs, alfa, beta, gamma):
        self.feedforward(inputs)
        self.compute_errors(outputs)
        self.update_weights(alfa, beta, gamma)


    def train_on_batch(self):
        pass


    def predict(self, X):
        return [self.feedforward(inputs) for inputs in X]


    def feedforward(self, inputs):
        outputs = self.layers[0].compute_output(inputs)
        for i in range(1, len(self.layers)) :
            outputs = self.layers[i].compute_output()

        return outputs


    def compute_errors(self, exp_outputs):
        # Computing output layer errors
        out_layer = self.layers[self.depth - 1]
        out_layer.errors = exp_outputs - out_layer.last_outputs

        # Computing hidden layers errors
        for i in range(self.depth-2, 0, -1) :
            self.layers[i].compute_errors()


    def update_weights(self, alfa, beta, gamma):
        for i in range(self.depth-1, 0, -1) :
            self.layers[i].update_weights(alfa, beta, gamma)


    def add_layer(self, layer):
        self.layers.append(layer)
        self.depth += 1


    def add_flatten_layer(self):
        prev_layer = self.layers[self.depth - 1]
        input_shape = prev_layer.last_outputs.shape

        flatten_layer = FlattenLayer(input_shape)

        prev_layer.next_layer = flatten_layer
        flatten_layer.prev_layer = prev_layer

        self.add_layer(flatten_layer)


    def add_dense_layer(self, dense_layer):
        prev_layer = self.layers[self.depth - 1]
        prev_layer.next_layers.append(dense_layer)
        dense_layer.prev_layer = prev_layer
        self.add_layer(dense_layer)


    def add_convolutional_layer(self, convolutional_layer):
        prev_layer = self.layers[self.depth - 1]
        prev_layer.next_layer = convolutional_layer
        convolutional_layer.prev_layer = prev_layer
        self.add_layer(convolutional_layer)


    def add_lstm_layer(self, lstm_layer):
        prev_layer = self.layers[self.depth - 1]

        if isinstance(prev_layer, LSTMInputLayer) :
            for unit, layer in zip(lstm_layer.units, prev_layer.layers) :
                unit.input_layer.prev_layer = layer
        elif isinstance(prev_layer, LSTMLayer) :
            for prev_unit, curr_unit in zip(prev_layer.units, lstm_layer.units):
                curr_unit.input_layer.prev_layer = prev_unit.hidden_layer

        prev_layer.next_layer = lstm_layer
        self.add_layer(lstm_layer)
