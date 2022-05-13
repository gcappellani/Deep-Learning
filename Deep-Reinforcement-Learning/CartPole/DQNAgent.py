import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras


class DQNAgent():
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.replay_buffer = deque(maxlen=3000)
        self.build_model()


    def loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))


    def sample_from_buffer(self, batch_size):
        sample_size = min(len(self.replay_buffer), batch_size)
        samples = random.choices(self.replay_buffer, k=sample_size)
        return map(list, zip(*samples))


    def build_model(self):
        inputs = keras.Input(shape=(self.state_size,))
        hidden = keras.layers.Dense(units=10, activation='tanh')(inputs)
        outputs = keras.layers.Dense(units=self.action_size)(hidden)
        self.q_table = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_table.compile(optimizer=optimizer, loss=self.loss)


    def get_action(self, state):
        q_state = self.q_table.predict(np.array([state]))[0]
        greedy_action = np.argmax(q_state)
        random_action = np.random.randint(self.action_size)

        return greedy_action if np.random.random() > self.eps else random_action


    def update_qtable(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.sample_from_buffer(batch_size=50)
        q_states = self.q_table.predict(np.array(states))
        q_next_states = self.q_table.predict(np.array(next_states))
        q_next_states[dones] = np.zeros([self.action_size])
        targets = rewards + self.discount_rate * np.max(q_next_states, axis=1)
        for i in range(len(q_states)) : q_states[i][actions[i]] = targets[i]

        self.q_table.fit(np.array(states), np.array(q_states), verbose=0)


    def train(self, episodes):
        episode_reward = 0
        for episode in range(episodes) :
            done = False
            state = self.env.reset()
            while not done :
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_qtable(state, action, next_state, reward, done)
                self.env.render()
                state = next_state
                episode_reward += reward

            print("Episode: {}, Episode reward: {}, eps: {}".format(episode, episode_reward, self.eps))
            episode_reward = 0
            self.eps *= .99

        print("Episode reward: {}".format(episode_reward))