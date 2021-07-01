# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:49:03 2021

@author: milo9
"""

import numpy as np
import tensorflow as tf
import time
from mycustomenvs import CartPoleEnv
from myRLtools import ER_Buffer
import pygame

#%% 8,56,60,61

env = CartPoleEnv()
n_traj = 1000
m_traj = 1
max_episodes = 200
mini_batch_size = 5000
gamma = 0.99
# tf.random.set_seed(33)
#%%
huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
#%%
num_obs = env.observation_space.shape[0]
num_actions = env.action_space.n  # 2

actor_input = tf.keras.Input(shape=(num_obs))
x = tf.keras.layers.Dense(32)(actor_input)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dense(num_actions, activation = 'softmax')(x)
ActorModel = tf.keras.Model(actor_input, x)

critic_input = tf.keras.Input(shape=(num_obs))
x = tf.keras.layers.Dense(32)(critic_input)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dense(1)(x)
CriticModel = tf.keras.Model(critic_input, x)

ActorOptimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
CriticOptimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)

#%% CREATE & FILL BUFFER    
ER = ER_Buffer(env,n_traj,m_traj,num_obs)
t = time.time()
ER.fill(ActorModel)
elapsed = time.time() - t
print(elapsed,'s')

#%%
curr_state = env.reset()    
live_running_reward = 0
t = time.time()
for i in range(max_episodes):
    print(i)
    for j in range(1):
        # ER.update(ActorModel)
        ER.update_batch(ActorModel)
    for k in range(10):
        
        # state_t1, action_t1, reward_t2, state_t2, done = ER.sample(mini_batch_size)
        
        state_t1 = 1.*ER.state
        action_t1 = 1.*ER.action
        # reward_t2 = 1.*ER.reward
        state_t2 = 1.*ER.next_state
        done = 1.*ER.done
        
        state_t1 = np.concatenate((state_t1,-state_t1),axis = 0)
        action_t1 = np.concatenate((action_t1,1.-action_t1),axis = 0)
        # reward_t2 = np.concatenate((reward_t2,reward_t2),axis = 0)
        state_t2 = np.concatenate((state_t2,-state_t2),axis = 0)
        done = np.concatenate((done,done),axis = 0)
        
        action_int = tf.cast(tf.reshape(action_t1, shape=(-1,1)),'int32')
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = ActorModel(state_t1, training=True)
            prob_act = tf.gather(probs, action_int, axis=1,batch_dims=1)
            values_t1 = CriticModel(state_t1, training=True)
            values_t2 = CriticModel(state_t2, training=True)
            r = tf.reshape(-1.*tf.cast(done, 'float32'),[-1,1]) + 0.*tf.reshape(tf.cast(state_t2[:,0], 'float32'),[-1,1])**2
            done_switch = tf.reshape(1.-tf.cast(done, 'float32'),[-1,1])
            returns = r + gamma*values_t2*done_switch
            advantage =  returns - values_t1
            log_probs = tf.math.log(prob_act)
            entropy_loss = -tf.reduce_mean(prob_act*log_probs)

            actor_loss = tf.reduce_mean(-log_probs*advantage) - 0.0001*entropy_loss
            critic_loss = huber(advantage,tf.zeros_like(advantage))
            
        grads1 = tape1.gradient(actor_loss, ActorModel.trainable_variables)
        grads2 = tape2.gradient(critic_loss, CriticModel.trainable_variables)
        ActorOptimizer.apply_gradients(zip(grads1, ActorModel.trainable_variables))
        CriticOptimizer.apply_gradients(zip(grads2, CriticModel.trainable_variables)) 
        
    # live_running_reward = 0
    # curr_state = env.reset()
    # for m in range(10000):
    #     action_probs = ActorModel(tf.expand_dims(curr_state,0))
    #     action = tf.math.argmax(action_probs,axis=1)[0].numpy()
    #     env_out = env.step(action,curr_state)
    #     curr_state = env_out[0]
    #     reward_metric = env_out[1]
    #     done_metric = env_out[2]
    #     live_running_reward = live_running_reward + reward_metric
    #     if done_metric:
    #         env.reset()
    #         break
    # print(f'\nSolved at episode {i}: running reward: {live_running_reward:.2f}!')

#     action_probs = ActorModel(tf.expand_dims(curr_state,0))
#     action = tf.math.argmax(action_probs,axis=1)[0].numpy()
#     env_out = env.step(action,curr_state)
#     curr_state = env_out[0]
#     reward = env_out[1]
#     done = env_out[2]
#     live_running_reward = live_running_reward + reward
#     if done:
#         curr_state = env.reset()
#         live_running_reward = 0
#     env.render()
#     print(f'\nSolved at episode {i}: running reward: {live_running_reward:.2f}!')
# env.close()   
eps = max_episodes/(time.time() - t)
print(eps)
#%%

pygame.joystick.quit() 
pygame.quit()
pygame.display.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

# ActorModel = tf.keras.models.load_model('saved_models\cartpole_6_6_21_a\Actor')
lockout_max = -1
lockout = lockout_max
running_reward = 0
curr_state = env.reset()
# curr_state = [0.0,3.0,180.0*np.pi/180,-7.5]
# curr_state = [2.3,0.0,-0.6531263,0.0]
rr = 0
while True:
    rr = rr+1
    action_probs = ActorModel(tf.expand_dims(curr_state,0))
    # val = CriticModel(tf.expand_dims(curr_state,0))
    # print(val)
    if joystick.get_button(0) and lockout<=0:
        action = 0
        lockout = lockout_max
    elif joystick.get_button(1) and lockout<=0:
        action = 1
        lockout = lockout_max
    else:
        lockout = lockout - 1
        action = tf.math.argmax(action_probs,axis=1)[0].numpy()
    env_out = env.step(action,curr_state)
    curr_state = env_out[0]
    reward = env_out[1]
    done = env_out[2]
    running_reward = running_reward + reward
    
    if done:
        curr_state = env.reset()
        print(rr)
        rr = 0
        # curr_state = [0.0,3.0,180.0*np.pi/180,-7.5]
    if joystick.get_button(3):
        env.reset()
        env.close()
        break
    time.sleep(0.00)    
    env.render()
env.close()
