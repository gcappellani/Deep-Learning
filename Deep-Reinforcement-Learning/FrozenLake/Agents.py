import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Agent():
    def __init__(self, env):
        self.env = env
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete

        if self.is_discrete :
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else :
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)
            print("Action shape:", self.action_shape)


    def get_action(self, state):
        if self.is_discrete :
            action = random.choice(range(self.action_size))
        else :
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)

        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()


    def build_model(self):
        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])


    def get_action(self, state):
        greedy_action = np.argmax(self.q_table[state])
        random_action = super().get_action(state)
        return greedy_action if np.random.random() > self.eps else random_action


    def update_qtable(self, state,  action, next_state, reward, done):
        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update


    def train(self, episodes):
        total_reward = 0
        for episode in range(episodes) :
            done = False
            state = self.env.reset()
            while not done :
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_qtable(state, action, next_state, reward, done)
                state = next_state
                total_reward += reward
                self.env.render()

            print("Episode: {}, Total reward: {}, eps: {}".format(episode, total_reward, self.eps))
            self.eps *= .99

        print("Total reward: {}".format(total_reward))


class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.x_replay_buffer = deque(maxlen=100)
        self.y_replay_buffer = deque(maxlen=100)
        self.build_model()


    def loss(self, y_true, y_pred):
        return tf.square(y_true - y_pred)


    def build_model(self):
        inputs = keras.Input(shape=(self.state_size,))
        outputs = keras.layers.Dense(units=self.action_size)(inputs)
        self.q_table = keras.Model(inputs=inputs, outputs=outputs)

        self.q_table.compile(optimizer='adam', loss=self.loss)


    def get_action(self, state):
        self.one_hot_state = tf.one_hot(state, depth=self.state_size).numpy()
        self.q_state = self.q_table.predict(self.one_hot_state.reshape(1,self.state_size))[0]
        greedy_action = np.argmax(self.q_state)
        random_action = super().get_action(state)

        return greedy_action if np.random.random() > self.eps else random_action


    def update_qtable(self, action, next_state, reward, done):
        one_hot_next_state = tf.one_hot(next_state, depth=self.state_size).numpy()
        q_next_state = self.q_table.predict(one_hot_next_state.reshape(1,self.state_size))
        if done : q_next_state = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next_state)
        self.q_state[action] = q_target

        self.x_replay_buffer.append(list(self.one_hot_state))
        self.y_replay_buffer.append(list(self.q_state))
        self.q_table.fit(np.array(list(self.x_replay_buffer)), np.array(list(self.y_replay_buffer)), verbose=0)


    def train(self, episodes):
        total_reward = 0
        for episode in range(episodes) :
            done = False
            state = self.env.reset()
            while not done :
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_qtable(action, next_state, reward, done)
                state = next_state
                total_reward += reward
                self.env.render()

            print("Episode: {}, Total reward: {}, eps: {}".format(episode, total_reward, self.eps))
            self.eps *= .99

        print("Total reward: {}".format(total_reward))

