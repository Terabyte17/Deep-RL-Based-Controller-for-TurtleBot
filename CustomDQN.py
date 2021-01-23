import gym
import os
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQN:
    def __init(self, env, MEMORY_SIZE=int(1e6), gamma=0.95, epsilon=0.95, epsilon_decay=0.95, epsilon_final=0.1, batch_size=32, lr=LR):
        self.env = env
        self.memory = deque(maxlen = MEMORY_SIZE)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final
        self.batch_size = batch_size
        self.LR = LR

        self.model = create_q_network()
        self.target_model = create_q_network()

    
    def create_q_network(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(self.env.observation_space.shape[0], ), activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_space.n, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=self.LR))
        return model
    
    def act(self, state):
        q_values = self.model.predict(state)
        policy = np.ones(q_values.shape[1], dtype=float)/q_values.shape[1] * self.epsilon
        best_action = np.argmax(q_values)
        policy[best_action]+=1-self.epsilon
        action = np.random.choice(np.arange(len(policy)), p=policy)
        return action
        
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory)<self.batch_size:
            return
        
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            target_values = self.target_model.predict(next_state)
            if done:
                q_update = reward
            else:
                q_update = reward + self.gamma * np.max(target_values)
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.epsilon*=self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_final)  

