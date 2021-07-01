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

from modified_envs import CartPoleEnv, AcrobotEnv
from agent import TDLambda

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed


n_step=5
t_step=1
lam = 0.0
min_episodes_criterion = 100
max_episodes = 50000
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = 'logs'
test_name = 'Cartpole_n{}t{}_tmp{}'.format(n_step,t_step,lam)
test_path = os.path.join(log_dir, test_name)
ckpt_path = os.path.join(test_path, 'ckpts')
train_log_path = os.path.join(test_path, 'training_log', current_time)
model_path = os.path.join(test_path, 'models')
video_path = os.path.join(test_path, 'videos')
print(video_path)
try:
    os.makedirs(test_path)
except OSError as error:    
    pass
    # print(error)
try:
    os.makedirs(ckpt_path)
except OSError as error:
    pass
    # print(error)
try:
    os.makedirs(train_log_path)
except OSError as error:
    pass
    # print(error)
try:
    os.makedirs(model_path)
except OSError as error:
    pass
    # print(error)
try:
    os.makedirs(video_path)
except OSError as error:
    pass
    # print(error)


# Set seed for experiment reproducibility
seed = 8675309
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

#%%

def make_env(rank, seed=0):
    def _init():
        env = CartPoleEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Create the environment
def create_environment(n_env):
    env = DummyVecEnv([make_env(i, seed=seed) for i in range(n_env)])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

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
start_time = time.time()

env, num_obs, num_actions = create_environment(n_env = 100)
eval_env = CartPoleEnv()
# eval_env = AcrobotEnv()
agent = TDLambda(num_actions, num_obs, n_steps=n_step, batch_size=100, num_env=env.num_envs, replay_buffer_size = 10000, ckpts_num=25, ckpt_dir=ckpt_path, lam=lam)
# agent.actor = tf.keras.models.load_model(model_path + '/actor_model')
agent.reset_experience_replay()
agent.fill_experience_replay(env)

e_time = time.time()-start_time
print('elapsed time = {}'.format(e_time))
#%%
# with tf.device('/cpu:0'):
with tf.device('/gpu:0'):
    
    start_time = time.time()
    
    reward_threshold = eval_env.max_steps-5
    running_reward = 0
    
    reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float32)
    summary_writer = tf.summary.create_file_writer(train_log_path)
    
    batch_mean_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
    tot_itr = 0
    episode_reward = 0
    with tqdm.trange(max_episodes) as t:
        with summary_writer.as_default():
            for i in t:
                agent.take_n_steps(env, 1)
                for _ in range(t_step):
                    #   agent.train()
                    agent.train_and_checkpoint(save_freq = 1000)
                
                if( i % 1000 == 0 ):
                    if(True):
                        episode_reward = 0
                        
                        state = eval_env.reset()
                        state = tf.expand_dims(state, 0)
                        for _ in range(1000):
                            action = agent.act(state, deterministic=True)
                            state, reward, done, _ = eval_env.step(action.numpy()[0])
                            state = tf.expand_dims(state, 0)
                            episode_reward += reward
                            eval_env.render()
                            if (tf.cast(done, tf.bool)):
                                break
                        eval_env.reset()
                        eval_env.close()
                        # print(episode_reward)
    
                        tf.summary.scalar('episode_reward', episode_reward, step=i)
                        # print(episode_reward)
                        
                if( i % 10 == 0 ):
                    tf.summary.scalar('actor_loss', agent.actor_loss_metric.result(), step=i)
                    agent.actor_loss_metric.reset_states()
                    batch_mean_end_reward = agent.total_reward_metric.result().numpy()
                    tf.summary.scalar('batch_mean_end_reward', batch_mean_end_reward, step=i)
                    
                    batch_mean_reward.append(batch_mean_end_reward)
                    agent.total_reward_metric.reset_states()
    
    
                running_reward = statistics.mean(batch_mean_reward)
    
                t.set_description(f'Episode {i}')
                t.set_postfix(batch_mean_end_reward=batch_mean_end_reward, episode_reward=episode_reward, running_reward=running_reward)
                tot_itr=i+1
                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    # break
                    pass
        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
    
    agent.actor.save(model_path + '/actor_model')
    
    e_time = time.time()-start_time
    print('elapsed time = {}'.format(e_time))
    
    file = open(test_path + '/info.txt', "w")
    text_too_write = 'runtime={}\ntotal_itr={}'.format(e_time, tot_itr)
    file.write(text_too_write)
    file.close()

#%%
print(agent.manager.checkpoints) 
#%%
agent = TDLambda(num_actions, num_obs, n_steps=n_step, batch_size=1000, num_env=env.num_envs, replay_buffer_size = 10000, ckpts_num=25, ckpt_dir=ckpt_path, lam=lam)
# agent.load_checkpoint(ckpt_path+'\ckpt-19')
agent.load_checkpoint()

for _ in range(1):
    eval_env = CartPoleEnv()
    episode_reward = 0
    
    state_tmp = [0, 0, 5*np.pi/180, 0.]
    eval_env.set_state(state_tmp)
    state = tf.constant(state_tmp)
    
    # state = eval_env.reset()
    state = tf.expand_dims(state, 0)
    
    logits = agent.actor(state)
    probs = tf.nn.softmax(logits)
    log_probs = tf.math.log(probs)
    for _ in range(400):
        action = agent.act(state, deterministic=True)
        state, reward, done, _ = eval_env.step(action.numpy()[0])
        state = tf.expand_dims(state, 0)
        episode_reward += reward
        eval_env.render()
        if (tf.cast(done, tf.bool)):
            break
    eval_env.reset()
    eval_env.close()
    print(episode_reward)

#%%

from PIL import Image
import moviepy.editor as mp
from datetime import datetime
    
def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int, angle: float, velocity: float): 
      screen = eval_env.render(mode='rgb_array')
      im = Image.fromarray(screen)
    
      images = [im]
    
      eval_env.reset()
      state_tmp = [0, velocity, angle*np.pi/180, 0.]
      eval_env.set_state(state_tmp)
      state = tf.constant(state_tmp)
      state = tf.expand_dims(state, 0)
      for i in range(1, 500 + 1):
    
        action = agent.act(state, deterministic=True)
        # action = agent.act(state, deterministic=False)
        state, reward, done, _ = eval_env.step(action.numpy()[0])
        state = tf.expand_dims(state, 0)
        # Render screen every n steps
        n=1
        if i % n == 0:
          screen = eval_env.render(mode='rgb_array')
          images.append(Image.fromarray(screen))
    
        if done:
          break
    
      return images

for v in [0, -0.8]: 
    for a in [35]:
        
        eval_env = CartPoleEnv()
        eval_env.reset()
        # screen = eval_env.render(mode='rgb_array')
        # print(screen)
    
        # Save GIF image
        images = render_episode(env, agent.actor, 100, angle=a, velocity=v)
        
        image_file = video_path + '/a{}_v{}.gif'.format(a, v)
        video_file = video_path + '/a{}_v{}.mp4'.format(a, v)
        # loop=0: loop forever, duration=1: play each frame for 1ms
        images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
        clip = mp.VideoFileClip(image_file)
        clip.write_videofile(video_file)
        eval_env.close()
        
