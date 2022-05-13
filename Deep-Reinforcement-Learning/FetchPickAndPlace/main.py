import gym
import numpy as np
from agent import Agent

import mujoco_py
import os

if __name__ == '__main__':
    env = gym.make('FetchReach-v1')

    '''
    env.reset()
    for _ in range(1000):
        action = np.random.uniform(-1, 1, (4,))
        state = env.step(action)
        env.render()
    '''

    agent = Agent(input_shape=(16,), action_dim=env.action_space.shape[0], env=env, noise=.2, fc1_dims=256, fc2_dims=256)
    n_episodes = 1000000

    evaluate = False
    eps = 1.
    for i in range(n_episodes):
        obs = env.reset()
        state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']], axis=0)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state, evaluate, eps)
            obs = env.step(action)
            new_state = np.concatenate([obs[0]['observation'], obs[0]['achieved_goal'], obs[0]['desired_goal']], axis=0)
            reward, done, info = obs[1], obs[2], obs[3]
            distance = np.linalg.norm(state[-6:-3] - state[-3:])
            if reward != 0:
                reward = distance / reward
            score += reward
            agent.remember(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            env.render()
        if eps > 0.0001:
            eps -= .0001

        print("Episode: {}, Total reward: {}, Epsilon: {}".format(i, score, eps))


