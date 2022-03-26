from DenseLayers import *


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
            for inputs, outputs in zip(X, Y) :
                self.train_on_item(inputs, outputs, alfa, beta, gamma)
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
        outputs = inputs
        for i in range(len(self.layers)) :
            outputs = self.layers[i].compute_output(outputs)

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


    def add_dense_input_layer(self, inputsize):
        self.layers.append(DenseInputLayer(inputsize))
        self.depth += 1


    def add_dense_layer(self, activation, size, weights=None, randrange=None, bias=.0):
        inputsize = self.layers[self.depth - 1].size
        dense_layer = DenseLayer(inputsize, activation, size, weights, randrange, bias)

        self.layers[self.depth - 1].next_layer = dense_layer
        dense_layer.prev_layer = self.layers[self.depth - 1]

        self.layers.append(dense_layer)
        self.depth += 1