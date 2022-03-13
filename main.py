from Faces import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ActivationFunctions import *
from NeuralNetwork import NeuralNetwork


if __name__ == '__main__':

    face_encoder = FaceEncoder()
    X, Y = face_encoder.get_working_set(img_scale='2', eye_state=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    nn = NeuralNetwork()
    nn.add_input_layer(len(X[0]))
    nn.add_dense_layer(activation=sigmoid, size=128, weights='random', randrange=(-.5, .5), bias=-.1)  # Hidden Layer
    nn.add_dense_layer(activation=sigmoid, size=2, weights='random', randrange=(-.5, .5), bias=-.1)    # Output Layer

    nn.train(X_train, Y_train, epochs=3, alfa=.1, beta=.1, gamma=.4)

    outputs = nn.predict(X_test)

    hypothesis = face_encoder.getHypothesis(outputs)

    print(accuracy_score(Y_test, hypothesis))
