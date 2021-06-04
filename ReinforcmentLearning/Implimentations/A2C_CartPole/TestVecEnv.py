# -*- coding: utf-8 -*-
"""
Created on Sun May 30 02:53:42 2021

@author: tylerbarkin
"""

import os
os.cpu_count()


import time
import datetime, os

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

from collections import namedtuple, deque
import random



# Set seed for experiment reproducibility
# seed = 66
seed = 8675302
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


env_id = "CartPole-v1"
num_cpu = 4  # Number of processes to use
# Create the vectorized environment
env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

# Box(4,) means that it is a Vector with 4 components
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape[0])
# Discrete(2) means that there is two discrete actions
print("Action space:", env.action_space)

import time
t = time.time()
for _ in range(1):
  obs = env.reset()
  actions = np.zeros( 4, dtype=int)
  env.step_async(actions)
  obs, reward, done, info = env.step_wait()

print(obs)

print('elapsed time = {}'.format(time.time()-t))


