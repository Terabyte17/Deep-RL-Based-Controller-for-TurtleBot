import os
import os.path as path
import sys
import gym
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import balance_bot
import tensorflow as tf 
import stable_baselines
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import CheckpointCallback

if __name__=="__main__":
    env = gym.make("balancebot-v0")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models',name_prefix='FirstModelForward')
    model = DQN.load("./prevmodel/FirstModel_160000_steps")
    model.set_env(env)
    model.exploration_initial_eps = 0.4
    model.learn(total_timesteps=int(1e5), tb_log_name="FirstRunForward", reset_num_timesteps=False, callback=checkpoint_callback)
    model.save("FirstModelFinalForward")
