import numpy 
import gym
import balance_bot
import time

if __name__=="__main__":
    env = gym.make("balancebot-PID-v0")
    obs = env.reset()
    error =  obs[0]
    kp = 10
    kd = 0.1
    error_derivative = obs[1]
    while True:
        action = (kp*-error)/2 + kd*-error_derivative
        print(action)
        obs, _, done, _ = env.step(action)
        error = obs[0]
        error_derivative = obs[1]
        if done:
            obs = env.reset()

