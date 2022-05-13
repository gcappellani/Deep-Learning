import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras


class QNetwork():
    def __init__(self, state_size, action_size, learning_rate):
        inputs = keras.Input(shape=(state_size,))
        hidden = keras.layers.Dense(units=10, activation='tanh')(inputs)
        outputs = keras.layers.Dense(units=action_size)(hidden)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')


    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit(x, y, verbose=0)

    def update_from(self, model, tau):
        curr_weights = self.model.get_weights()
        weights = model.get_weights()
        for cw, w in zip(curr_weights, weights) : cw = tau * w + (1-tau) * cw
        self.model.set_weights(curr_weights)


class DoubleDQNAgent():
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.replay_buffer = deque(maxlen=3000)
        self.online_qnn = QNetwork(self.state_size, self.action_size, self.learning_rate)
        self.offline_qnn = QNetwork(self.state_size, self.action_size, self.learning_rate)
        self.offline_qnn.model.set_weights(self.online_qnn.model.get_weights())


    def sample_from_buffer(self, batch_size):
        sample_size = min(len(self.replay_buffer), batch_size)
        samples = random.choices(self.replay_buffer, k=sample_size)
        return map(list, zip(*samples))


    def get_action(self, state):
        q_state = self.online_qnn.predict(np.array([state]))[0]
        greedy_action = np.argmax(q_state)
        random_action = np.random.randint(self.action_size)

        return greedy_action if np.random.random() > self.eps else random_action


    def update_qnetwork(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.sample_from_buffer(batch_size=50)

        on_q_states = self.online_qnn.predict(np.array(states))

        on_q_next_states = self.online_qnn.predict(np.array(next_states))
        on_q_next_states[dones] = np.zeros([self.action_size])
        best_actions = np.argmax(on_q_next_states, axis=1)

        off_q_next_states = self.offline_qnn.predict(np.array(next_states))
        action_values = np.array([off_q_next_state[action] for off_q_next_state, action in zip(off_q_next_states, best_actions)])
        targets = rewards + self.discount_rate * action_values

        for i in range(len(on_q_states)) : on_q_states[i][actions[i]] = targets[i]

        self.online_qnn.fit(np.array(states), np.array(on_q_states))
        #self.offline_qnn.model.set_weights(self.online_qnn.model.get_weights())


    def train(self, episodes):
        episode_reward = 0
        update_counter = 1
        for episode in range(episodes) :
            done = False
            state = self.env.reset()
            while not done :
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_qnetwork(state, action, next_state, reward, done)
                if update_counter % 25 == 0 : self.offline_qnn.update_from(self.online_qnn.model, tau=0.01)

                self.env.render()
                state = next_state
                update_counter += 1
                episode_reward += reward

            print("Episode: {}, Episode reward: {}, eps: {}".format(episode, episode_reward, self.eps))
            episode_reward = 0
            self.eps *= .99

        print("Episode reward: {}".format(episode_reward))