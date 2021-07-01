
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:18:38 2021

@author: milo9
"""

import numpy as np
import tensorflow as tf

class ER_Buffer():
    def __init__(self,env,n_traj,traj_per_buffer,num_obs):
        self.env = env
        self.n_traj = n_traj
        self.traj_per_buffer = traj_per_buffer
        self.num_trans = self.n_traj*self.traj_per_buffer
        self.num_obs = num_obs
        self.state = np.zeros([self.num_trans,self.num_obs],dtype= float)
        self.action = np.zeros([self.num_trans,],dtype = float)
        self.reward = np.zeros([self.num_trans,],dtype = float)
        self.next_state = np.zeros([self.num_trans,self.num_obs],dtype= float)
        self.done = np.zeros([self.num_trans,],dtype= float)
        self.idx = 0
    def fill(self,ActorModel):
        self.curr_states = np.zeros([self.n_traj,self.num_obs],dtype= float)
        self.next_states = np.zeros([self.n_traj,self.num_obs],dtype= float)
        self.rewards = np.zeros([self.n_traj,],dtype= float)
        self.dones = np.zeros([self.n_traj,],dtype= float)
        
        for j in range(self.n_traj):
            self.curr_states[j,:] = self.env.reset()
        
        count = 0    
        
        for i in range(self.traj_per_buffer):
            
            self.probs = ActorModel(self.curr_states.astype('float32'))
            # self.actions = np.array(tf.random.categorical(self.probs,1)[:,0])
            self.actions = np.array(tf.argmax(self.probs,axis = 1))
            # self.random_actions = 
            
            
            for j in range(self.n_traj):
                
                env_out = self.env.step(self.actions[j],self.curr_states[j,:])
                self.next_states[j,:] = env_out[0]
                self.rewards[j] = env_out[1]
                self.dones[j] = env_out[2]
                
                if self.dones[j]==1.:
                    self.next_states[j,:] = self.env.reset()
                 
                self.state[count,:] = self.curr_states[j,:]
                self.next_state[count,:] = self.next_states[j,:] 
                self.done[count] = self.dones[j]
                self.reward[count] = self.rewards[j]
                self.action[count] = self.actions[j]
                
                count = count+1
            self.curr_states = 1*self.next_states
            
    def update(self,ActorModel):
        # idx = np.random.randint(0, high=self.num_trans, size=[self.n_traj,], dtype=int)
        self.curr_states = 1.*self.next_states
        self.probs = ActorModel(self.curr_states.astype('float32'))
        # self.actions = np.array(tf.random.categorical(self.probs,1)[:,0])
        self.actions = np.array(tf.argmax(self.probs,axis = 1))
        # print(self.actions)
        if self.idx == self.traj_per_buffer:
            self.idx = 0
        
        
        for j in range(self.n_traj):
            
            env_out = self.env.step(self.actions[j],self.curr_states[j,:])
            self.next_states[j,:] = env_out[0]
            self.rewards[j] = env_out[1]
            self.dones[j] = env_out[2]
            
            if self.dones[j]==1.:
                self.next_states[j,:] = self.env.reset()

        self.state[self.idx*self.n_traj:(self.idx+1)*self.n_traj,:] = self.curr_states
        self.action[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.actions
        self.reward[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.rewards
        self.next_state[self.idx*self.n_traj:(self.idx+1)*self.n_traj,:] = self.next_states
        self.done[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.dones
        
        self.idx = 1+self.idx
    
    def update_batch(self,ActorModel):
        
        self.curr_states = 1.*self.next_states
        self.probs = ActorModel(self.curr_states.astype('float32'))
        # self.actions = np.array(tf.random.categorical(self.probs,1)[:,0])
        self.actions = np.array(tf.argmax(self.probs,axis = 1))
        # print(self.actions)
        if self.idx == self.traj_per_buffer:
            self.idx = 0
        
        env_out = self.env.steps(self.actions,self.curr_states)
        
        self.next_states = env_out[0]
        self.rewards = env_out[1]
        self.dones = env_out[2]

        dones_bool = self.dones == 1.
        n_dones = np.sum(self.dones).astype(int)
        reset_states = np.random.uniform(low=-0.05, high=0.05, size=(n_dones,4))
        self.next_states[dones_bool,:] = reset_states;

        self.state[self.idx*self.n_traj:(self.idx+1)*self.n_traj,:] = self.curr_states
        self.action[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.actions
        self.reward[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.rewards
        self.next_state[self.idx*self.n_traj:(self.idx+1)*self.n_traj,:] = self.next_states
        self.done[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.dones
        
        self.idx = 1+self.idx
        
        
    def sample(self,num_samples):
        idx = np.random.randint(0, high=self.num_trans, size=[num_samples,], dtype=int)
        return self.state[idx,:],self.action[idx,],self.reward[idx,],self.next_state[idx,:],self.done[idx,]