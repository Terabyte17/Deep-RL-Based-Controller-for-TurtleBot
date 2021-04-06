import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

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
    model = DQN.load("dqn_tensorboard/FirstModel_160000_steps")
    obs = env.reset()
    vt = []
    while True:
        action = model.predict(obs)
        print(action)
        vt.append(env.vt)
        obs, reward, done, _ = env.step(action[0])
        if done:
            obs = env.reset()
            break
    plt.plot(vt)
    plt.xlabel("Timesteps")
    plt.ylabel("Velocity of right wheel")
    plt.title("Model Analysis")
    plt.savefig("Model Analysis")