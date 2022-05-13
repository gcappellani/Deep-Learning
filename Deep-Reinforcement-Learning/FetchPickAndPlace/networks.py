import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=128, name='critic', checkpoint_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.fc1 = Dense(self.fc2_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, action_dim, fc1_dims=128, fc2_dims=128, name='actor', checkpoint_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dim = action_dim

        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.fc1 = Dense(self.fc2_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.a = Dense(self.action_dim, activation='tanh')

    def call(self, state):
        action = self.fc1(state)
        action = self.fc2(action)

        a = self.a(action)

        return a
