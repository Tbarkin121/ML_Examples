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

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# Set seed for experiment reproducibility
seed = 66
# seed = 8675309
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

#%%
def make_env(rank, seed=0):
    def _init():
        env = CartPoleEnv()
        # env.set_target(target=1.0, weight=0.2)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


# Create the environment

def create_environment(n_env):

    env = DummyVecEnv([make_env(i, seed=seed) for i in range(n_env)])
    

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
#%%

env, num_obs, num_actions = create_environment(n_env = 100)
eval_env = CartPoleEnv()

agent = ACER(num_actions, num_obs, batch_size=10000, num_env=env.num_envs, replay_buffer_size = 100000)
agent.reset_experience_replay()
#%%
t = time.time()
agent.fill_experience_replay(env)
print('elapsed time = {}'.format(time.time()-t))


min_episodes_criterion = 100
max_episodes = 100000
episode_length = 10000
reward_threshold = episode_length-5
running_reward = 0

reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float32)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/gradient_tape/' + current_time + '/grand'
summary_writer = tf.summary.create_file_writer(log_dir)

episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
with tqdm.trange(max_episodes) as t:
    with summary_writer.as_default():
        for i in t:
            agent.take_n_steps(env, 20)
            for _ in range(10):
                  agent.train()
                 
            # Run an episode to test
            if i % 10 == 0:
                tf.summary.scalar('actor_loss', agent.actor_loss_metric.result(), step=i)
                tf.summary.scalar('critic_loss', agent.critic_loss_metric.result(), step=i)
                # tf.summary.scalar('logits1', agent.logits1_metric.result(), step=i)
                # tf.summary.scalar('logits2', agent.logits2_metric.result(), step=i)
                # tf.summary.scalar('probs1', agent.probs1_metric.result(), step=i)
                # tf.summary.scalar('probs2', agent.probs2_metric.result(), step=i)
                agent.actor_loss_metric.reset_states()
                agent.critic_loss_metric.reset_states()
                # agent.logits1_metric.reset_states()
                # agent.logits2_metric.reset_states()
                # agent.probs1_metric.reset_states()
                # agent.probs2_metric.reset_states()
        
                episode_reward = 0
                state = tf.constant(eval_env.reset(), dtype=tf.float32)
                state = tf.expand_dims(state, 0)
                logits = agent.actor(state)
                probs = tf.nn.softmax(logits)
                log_probs = tf.math.log(probs)
                for _ in range(episode_length):
                    action = agent.act(state, deterministic=True)
                    state, reward, done, _ = eval_env.step(action.numpy())
                    state = tf.expand_dims(state, 0)
                    episode_reward += reward
                    if (tf.cast(done, tf.bool)):
                        eval_env.reset()
                        break
                reward_metric(episode_reward)
                tf.summary.scalar('episode reward', reward_metric.result(), step=i)
                reward_metric.reset_states()
    
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)
                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            if running_reward > reward_threshold and i >= min_episodes_criterion:  
              break
    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
    
agent.actor.save('logs/models/actor_model')
agent.critic.save('logs/models/critic_model')
# def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#   """Returns state, reward and done flag given an action."""

#   state, reward, done, _ = env.step(action)
#   return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))


# def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
#   return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])


#%%
agent = ACER(num_actions, num_obs, batch_size=1000, num_env=env.num_envs, replay_buffer_size = 10000)
agent.actor = tf.keras.models.load_model('logs/models/actor_model')
agent.critic = tf.keras.models.load_model('logs/models/critic_model')

#%%
eval_env = CartPoleEnv()
episode_reward = 0
state_tmp = [0., 0., 25.*np.pi/180, 0.]
eval_env.set_state(state_tmp)
state = tf.constant(state_tmp)
state = tf.expand_dims(state, 0)


logits = agent.actor(state)
probs = tf.nn.softmax(logits)
log_probs = tf.math.log(probs)
for _ in range(500):
    action = agent.act(state, deterministic=True)
    state, reward, done, _ = eval_env.step(action.numpy())
    state = tf.expand_dims(state, 0)
    episode_reward += reward
    eval_env.render()
    if (tf.cast(done, tf.bool)):
        break
eval_env.reset()
eval_env.close()
print(episode_reward)

#%%
agent.reset_envs(env)
agent.take_n_steps(env, 100)
print(agent.done)