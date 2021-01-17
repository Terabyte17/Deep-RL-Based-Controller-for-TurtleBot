import numpy 
import gym
import balance_bot
import time

if __name__=="__main__":
    env = gym.make("balancebot-v0")
    env.reset()
    i = 0
    while True:
        env.step(4)
