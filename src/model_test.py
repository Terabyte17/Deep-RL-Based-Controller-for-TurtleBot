import os
import sys
import gym
import numpy as np
import pybullet as p
import time
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
    model = DQN.load("prevmodel/FirstModel_160000_steps")
    obs = env.reset()
    p.resetDebugVisualizerCamera(3, 140, -10, [0, 0, 0.1])
    time.sleep(10000)
    time.sleep(10)
    vt = []
    while True:
        action = model.predict(obs)
        print(action)
        vt.append(env.vt)
        obs, reward, done, _ = env.step(action[0])
        if done:
            obs = env.reset()
    plt.plot(vt)
    plt.xlabel("Timesteps")
    plt.ylabel("Velocity of right wheel")
    plt.title("Model Analysis")
    plt.savefig("Model Analysis")