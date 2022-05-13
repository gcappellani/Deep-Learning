import gym
from Agents import *
from gym.envs.registration import register

if __name__ == '__main__':

    try:
        register(
            id='FrozenLakeNoSlip-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.78,  # optimum = .8196
        )
    except:
        pass

    env_name = "FrozenLakeNoSlip-v0"
    env = gym.make(env_name)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    agent = QNAgent(env)
    agent.train(100)
    agent.train(100)
    agent.train(100)
