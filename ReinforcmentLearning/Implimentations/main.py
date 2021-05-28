#%%
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

from modified_envs import CartPoleEnv
from agent import ACER

# Set seed for experiment reproducibility
# seed = 66
seed = 8675302
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

#%%
# Create the environment
def create_environment():
    env = CartPoleEnv()
    env.set_target(target=1.0, weight=0.2)
    env.seed(seed)
    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # Box(4,) means that it is a Vector with 4 components
    print("Observation space:", env.observation_space)
    print("Observation Shape:", env.observation_space.shape[0])
    # Discrete(2) means that there is two discrete actions
    print("Action space:", env.action_space)

    print('num_obs = {}'.format(num_obs))
    print('num_actions = {}'.format(num_actions))

    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    return env, num_obs, num_actions

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])
  

env, num_obs, num_actions = create_environment()

#%%
agent = ACER(env, num_actions, mem_size=3, batch_size=100, mini_batch_size=10, n_mini_batches=10)
agent.reset_replay_buffer()
agent.fill_replay_buff(3)

state = tf.constant(env.reset(), dtype=tf.float32)
state = tf.expand_dims(state, 0)
print(state)