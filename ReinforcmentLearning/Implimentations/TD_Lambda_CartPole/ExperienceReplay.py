import numpy as np
import tensorflow as tf
from collections import namedtuple, deque

class ER_Buffer():
    def __init__(self, env, n_traj, traj_per_buffer, num_obs):
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
            logits = ActorModel(self.curr_states.astype('float32'))
            probs = tf.nn.softmax(logits)
            log_probs = tf.math.log(probs)
            self.actions = np.array(tf.random.categorical(log_probs,1)[:,0])
            
            for j in range(self.n_traj):
                env_out = self.env.step(self.actions[j],self.curr_states[j,:])
                self.next_states[j,:] = env_out[0]
                self.rewards[j] = env_out[1]
                self.dones[j] = env_out[2]
                # self.env.reset()
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
        # self.curr_states = self.next_state[-self.n_traj::,:]
        self.curr_states = self.next_states.copy()

        logits = ActorModel(self.curr_states.astype('float32'))
        probs = tf.nn.softmax(logits)
        log_probs = tf.math.log(probs)
        self.actions = np.array(tf.random.categorical(log_probs,1)[:,0])
        
        for j in range(self.n_traj):
            env_out = self.env.step(self.actions[j],self.curr_states[j,:])
            self.next_states[j,:] = env_out[0]
            self.rewards[j] = env_out[1]
            self.dones[j] = env_out[2]
            # self.env.reset()
            if self.dones[j]==1.:
                self.next_states[j,:] = self.env.reset()

            
        self.state[self.idx*self.n_traj:(self.idx+1)*self.n_traj,:] = self.next_states.copy()
        self.action[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.actions.copy()
        self.reward[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.rewards.copy()
        self.next_state[self.idx*self.n_traj:(self.idx+1)*self.n_traj,:] = self.next_states.copy()
        self.done[self.idx*self.n_traj:(self.idx+1)*self.n_traj,] = self.dones.copy()
        
        self.idx = 1+self.idx
        if self.idx == self.traj_per_buffer:
            self.idx = 0
        
        
        
    def sample(self,num_samples):
        idx = np.random.randint(0, high=self.num_trans, size=[num_samples,], dtype=int)
        return self.state[idx,:], tf.expand_dims(self.action[idx,], 0), tf.expand_dims(self.reward[idx,], 0),self.next_state[idx,:],tf.expand_dims(self.done[idx,], 0)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ExperienceReplay(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#This is the old fill buffer code

        # # Reset Environment
        # state = tf.constant(env.reset(), dtype=tf.float32)
        # # Convert state into a batched tensor (batch size = 1)
        # state = tf.expand_dims(state, 0)
        # for i in range(n_samples):
        #   # Run the model and to get action probabilities and critic value
        #   action = self.act(state)

        #   # print('action = {}'.format(action))
        #   # Apply action to the environment to get next state and reward
        #   next_state, reward, done = tf_env_step(action)
        #   next_state = tf.expand_dims(next_state, 0)
        
        #   # print('next_state = {}'.format(next_state))
        #   # print('reward = {}'.format(reward))
        #   # print('done = {}'.format(done))
        #   # Store the transition in memory
        #   # print('')
        #   #If Episode is done, reset the environment
        #   if tf.cast(done, tf.bool):
        #     self.memory.push(state, action, next_state, 0, done)
        #     state = tf.constant(env.reset(), dtype=tf.float32)
        #     state = tf.expand_dims(state, 0)
        #   else:
        #     self.memory.push(state, action, next_state, reward, done)
        #     state=next_state
