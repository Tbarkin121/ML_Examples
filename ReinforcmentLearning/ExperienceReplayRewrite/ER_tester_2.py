# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:15:59 2021

@author: milo9
"""

#%% PACKAGES
import time
from mycustomenvs import CartPoleEnv
from myRLtools import ER_Buffer
import tensorflow as tf

tf.keras.backend.set_floatx('float32')
#%% INPUT
env = CartPoleEnv()
max_iter = 501
n_traj = 500
m_traj = 500
num_samples = 1000

#%% SETUP
num_obs = env.observation_space.shape[0]
num_actions = env.action_space.n

#%% DEFINE MODEL
actor_input = tf.keras.Input(shape=(num_obs))
x = tf.keras.layers.Dense(32)(actor_input)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dense(num_actions, activation = 'softmax')(x)
ActorModel = tf.keras.Model(actor_input, x)
ActorModel.compile(optimizer="adam", loss="categorical_crossentropy")

#%% CREATE & FILL BUFFER    
ER = ER_Buffer(env,n_traj,m_traj,num_obs)
t = time.time()
ER.fill(ActorModel)
elapsed = time.time() - t
print(elapsed,'s')
#%% TEST UPDATE
t = time.time()
for _ in range(max_iter):
   print(_)
   ER.update(ActorModel)
   state, action, reward, next_state, done = ER.sample(num_samples)
    
elapsed = time.time() - t
print(elapsed,'s')
print(n_traj*max_iter/elapsed,'fps')

