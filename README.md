# Controller for Self Balancing and Locomotion of TurtleBot
## Introduction:
One of the key challenges in robotics and control systems, is the development of a robust 
controller capable of handling perturbations in the surrounding environment especially 
for systems which are inherently unstable. However, in recent years model-free Deep 
Reinforcement Learning based methods have proven to be quite successful in building 
robust enough controllers without having to model the perturbations in the environment. 
In this work, we aim to develop a controller capable of both self-balancing a 2 wheeled 
robot, also known as TurtleBot, and making it move in its environment by modelling it as 
a Deep RL optimization problem.

## Installation Instructions:
Install the required python packages by running the following command:
~~~bash
pip install -r requirements.txt
~~~
Then to install the gym environment run, go to the balancebot folder and run:
~~~bash
pip install -e .
~~~

## Instantiating the environment
The gym environment can be instantiated as follows:
<br>
`import gym`
<br>
`import balance_bot`
<br>
`env = gym.make("balancebot-v0")`
<br>
The code base was tested with gym (0.18.0), pybullet (3.0.8), stable-baselines (2.10.1) with a python version of 3.7.6. However it is expected to work fine for any future versions of these packages, though they haven't been tested. The custom algorithm is compatible with Tensorflow 2+ version, but the Stable Baselines version is only compatible with versions<1.5.


