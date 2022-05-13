import gym
from DQNAgent import *
from DoubleDQNAgent import *

if __name__ == '__main__':

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    agent = DoubleDQNAgent(env)
    agent.train(100)
    agent.train(100)
    agent.train(100)
