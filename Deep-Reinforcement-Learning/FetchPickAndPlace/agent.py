import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from replaybuffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork

class Agent:
    def __init__(self, input_shape, action_dim, actor_lr=0.001, critic_lr=0.002, env=None, gamma=0.99,
                 buffer_size=100000, tau=0.005, fc1_dims=128, fc2_dims=128, batch_size=64, noise=0.1):
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.tau = tau
        self.batch_size = batch_size
        self.tau = tau
        self.noise = noise

        self.memory = ReplayBuffer(max_size=buffer_size, input_shape=input_shape, action_dim=action_dim)
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(action_dim=action_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_actor = ActorNetwork(action_dim=action_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_actor')
        self.target_critic = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr)) # lr not used, we update using actor params
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr)) # lr not used, we update using critic params

    def update_network_parameters(self, tau=1):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False, eps=0):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor.call(state)
        if not evaluate and np.random.random() < eps:
            actions = np.random.uniform(-1, 1, (1, self.action_dim))
            #actions += tf.random.normal(shape=[self.action_dim], mean=0.0, stddev=self.noise)
        #actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_next_actions = self.target_actor.call(new_states)
            target_next_values = tf.squeeze(self.target_critic.call(new_states, target_next_actions), 1)

            curr_values = tf.squeeze(self.critic.call(states, actions), 1)
            target = rewards + self.gamma * target_next_values * (1 - dones)
            critic_loss = keras.losses.MSE(target, curr_values)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor.call(states)
            actor_loss = -self.critic.call(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
