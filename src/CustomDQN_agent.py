import os
import sys
import gym
import random
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import balance_bot
import tensorflow as tf
from tensorflow.keras.models import load_model
from CustomDQN import DQN
import datetime

if __name__=="__main__":
    env = gym.make("balancebot-v0")
    dqn_agent = DQN(env)
    num_episodes = 1000
    
    train_log_dir = './customDQN'
    #train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        state = state.reshape((1,3))
        action = dqn_agent.act(state)
        episode = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape((1,3))
            next_action = dqn_agent.act(next_state)
            dqn_agent.add_experience(state, action, reward, next_state, done)
            episode_rewards.append(reward)
            if done is True:
                episode+=1
                state = env.reset()
                state = state.reshape((1,3))
                action = dqn_agent.act(state)
                #with train_summary_writer.as_default():
                    #ep_reward = np.sum(np.array(episode_rewards))
                    #tensor_reward = tf.constant(ep_reward)
                    #tf.summary.scalar('total reward', tensor_reward, step=episode)
            state = next_state
            action = next_action
            dqn_agent.experience_replay()

