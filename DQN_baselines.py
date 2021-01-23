import os
import gym
import numpy as np

import balance_bot
import tensorflow as tf 
import stable_baselines
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

if __name__=="__main__":
    env = gym.make("balancebot-v0")
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1, tensorboard_log="./dqn_tensorboard/")
    model.learn(total_timesteps=int(10), tb_log_name="fir_run", reset_num_timesteps=False)
    model.save("first_model")
