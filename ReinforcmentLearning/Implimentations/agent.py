import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from ExperienceReplay import ER_Buffer
from collections import deque

from ReplayBuffers import ExperienceReplay
import time

import numpy as np

class ActorNet(tf.keras.Model):
  """Actor network."""
  def __init__(self, num_actions: int, num_hidden_units: int):
    """Initialize."""
    super().__init__()
    # self.actor_input = layers.Input(shape=(4))
    self.d1 = layers.Dense(num_hidden_units)
    self.lr1 = layers.LeakyReLU()
    self.d2 = layers.Dense(num_hidden_units)
    self.lr2 = layers.LeakyReLU()
    # self.a = layers.Dense(num_actions, activation='softmax')
    self.a = layers.Dense(num_actions, activation='tanh')
    # self.a = layers.Dense(num_actions, activation='sigmoid')
    # self.a = layers.Dense(num_actions)
    

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    # x = tf.keras.activations.tanh(x)
    x = self.lr1(x)
    x = self.d2(x)
    x = self.lr2(x)
    return self.a(x)


class CriticNet(tf.keras.Model):
  """Critic network."""
  def __init__(self, num_actions:int, num_hidden_units: int):
    """Initialize."""
    super().__init__()
    # self.critic_input = layers.Input(shape=(4))
    self.d1 = layers.Dense(num_hidden_units)
    self.lr1 = layers.LeakyReLU()
    self.d2 = layers.Dense(num_hidden_units)
    self.lr2 = layers.LeakyReLU()
    # self.critic = layers.Dense(num_actions)
    self.c = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    # x = tf.keras.activations.tanh(x)
    x = self.lr1(x)
    x = self.d2(x)
    x = self.lr2(x)
    return self.c(x)

class ACER():
    def __init__(self, num_actions, num_obs, gamma = 0.99, batch_size = 128, n_steps=500, num_env=1, replay_buffer_size = 10000):
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_env=num_env

        #Setting up Experience Replay Buffer
        self.traj_length = tf.cast(replay_buffer_size/num_env, tf.int64)
        self.memory = ExperienceReplay(num_env, self.traj_length)

        self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.actor = ActorNet(self.num_actions, 64)
        self.critic = CriticNet(self.num_actions, 64)

        self.actor.compile(
                    optimizer='adam')

        self.critic.compile(
                    optimizer='adam')
        
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.I = 1
        # Define our metrics
        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        self.critic_loss_metric = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
        self.logits1_metric = tf.keras.metrics.Mean('logits1', dtype=tf.float32)
        self.logits2_metric = tf.keras.metrics.Mean('logits2', dtype=tf.float32)
        self.probs1_metric = tf.keras.metrics.Mean('probs1', dtype=tf.float32)
        self.probs2_metric = tf.keras.metrics.Mean('probs2', dtype=tf.float32)
        
        # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
          
    def act(self, state, deterministic=False):
        logits = self.actor(state)
        if (deterministic):
            action = tf.argmax(logits, 1)[0]
        else:
            # action = tf.random.categorical(logits, 1)[0, 0]
            # print('logits a = {}'.format(logits))
            # logits = logits + 0.1
            # print('logits b = {}'.format(logits))
            probs = tf.nn.softmax(logits)
            log_probs = tf.math.log(probs)
            action = tf.random.categorical(log_probs, 1)[:,0]
            # print('logits = {}'.format(logits))
            # print('probs = {}'.format(probs))
            # print('log_probs = {}'.format(log_probs))
            # print('action = {}'.format(action))
        return action

    def reset_experience_replay(self):
        print('Resetting Experience Replay Buffer')
        self.memory.replay_buffer.clear()
        # pass
    
    def fill_experience_replay(self, env):
        print('Filling Experience Replay Buffer')
        self.reset_envs(env)
        self.take_n_steps(env, n_steps = self.traj_length)
    
    def reset_envs(self, env):
        self.state_t1 = env.reset()

    def take_n_steps(self, env, n_steps=1):
        # print('Filling Experience Replay Buffer')
        # print(env.num_envs)
        
        for _ in range(n_steps):
            # action_t1_test = np.zeros(self.num_env, dtype=int)
            self.action_t1 = self.act(self.state_t1, deterministic=False)
            env.step_async(self.action_t1.numpy())
            self.state_t2, self.reward_t2, self.done, self.info = env.step_wait()
            self.action_t1 = tf.cast(tf.reshape(self.action_t1, shape=(self.num_env,1)),tf.int32)
            self.reward_t2 = tf.reshape(self.reward_t2, shape=(self.num_env,1))
            self.done = tf.reshape(self.done, shape=(self.num_env,1))
            values = (self.state_t1, self.action_t1, self.reward_t2, self.state_t2, self.done)
            values_batched = tf.nest.map_structure(lambda t: tf.stack(t), values)
            self.memory.replay_buffer.add_batch(values_batched)
            self.state_t1 = self.state_t2

    # @tf.function
    def train(self):
        # print('Running 1 Training Step')

        # Read all elements in the replay buffer:
        # print('~~~~~~~~~')
        # print('~~~~~~~~~')
        # trajectories = self.memory.replay_buffer.gather_all()
        # print("Trajectories from gather all:")
        # print(tf.nest.map_structure(lambda t: t.shape, trajectories))
        # print(trajectories)
        # # print(trajectories[3])
        # print(trajectories[0][:,0,:]) #First index is the data type, second is [entry in batch, batch number, data number]
        # print(trajectories[3][:,0,:]) #First index is the data type, second is [entry in batch, batch number, data number]

        # Get one sample from the replay buffer with batch size (self.batch_size) and 1 timestep:
        sample = self.memory.replay_buffer.get_next(sample_batch_size=self.batch_size, num_steps=1)
        # print('~~~~~~~~~')
        # print(sample)
        
        state_t1 = tf.squeeze(sample[0][0], axis=1)
        action_t1 = tf.squeeze(sample[0][1], axis=1)
        reward_t2 = tf.squeeze(sample[0][2], axis=1)
        state_t2 = tf.squeeze(sample[0][3], axis=1)
        done = tf.squeeze(sample[0][4], axis=1)
        # print('state_t1 = {}'.format(state_t1))
        # print('action_t1 = {}'.format(action_t1))
        # print('reward_t2 = {}'.format(reward_t2))
        # print('state_t2 = {}'.format(state_t2))
        # print('done = {}'.format(done))
        
        
        # mini_batch = list(agent.memory.memory)[0:self.mini_batch_size]
        # mini_batch = Transition(*zip(*mini_batch))
        # state_t1 = tf.reshape(mini_batch[0], shape=(self.mini_batch_size, num_obs))
        # action_t1 = tf.reshape(mini_batch[1], shape=(self.mini_batch_size, 1))
        # state_t2 = tf.reshape(mini_batch[2], shape=(self.mini_batch_size, num_obs))
        # reward_t2 = tf.reshape(mini_batch[3], shape=(self.mini_batch_size, 1))
        # done = tf.reshape(mini_batch[4], shape=(self.mini_batch_size, 1))
        

    
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            logits = self.actor(state_t1, training=True)
            values_t1 = self.critic(state_t1, training=True)
            values_t2 = self.critic(state_t2, training=True)
            # print('!!!!!!!!!!!!!!!!!!!!!!')
            # print('logits = {}'.format(logits))
 
            probs = tf.nn.softmax(logits)
            # print('probs = {}'.format(probs))
            # Fix Logging Later... something is wrong with the indicies
            # l = logits.numpy()
            # p = probs.numpy()
            # self.logits1_metric(l[0])
            # self.logits2_metric(l[1])
            # self.probs1_metric(p[0])
            # self.probs2_metric(p[1])
            
            probs_actions = tf.gather(probs, action_t1, axis=1, batch_dims=1)
            log_probs = tf.math.log(probs_actions)
            # print('action_t1 = {}'.format(action_t1))
            # print('probs_actions = {}'.format(probs_actions))
            # print('probs_actions.shape = {}'.format(probs_actions.shape))
            # print('log_probs = {}'.format(log_probs))
            # print('log_probs.shape = {}'.format(log_probs.shape))
            # time.sleep(5)
            
            
            returns = (tf.cast(reward_t2, 'float32') + self.gamma*values_t2)*(1-tf.cast(done, 'float32'))
            # returns = -tf.cast(done, 'float32') + self.gamma*values_t2*(1-tf.cast(done, 'float32'))
            advantage =  returns - values_t1 

            entropy_loss = -tf.math.reduce_mean(probs_actions*log_probs)
            # actor_loss = -log_probs*advantage
            # actor_loss = tf.math.reduce_mean(-self.I*log_probs * advantage) - 0.00001*entropy_loss

            actor_loss = tf.math.reduce_mean(-log_probs * advantage) - 0.00001*entropy_loss
            # self.I *= self.gamma
            # critic_loss = 0.5*self.huber_loss(values_t1, returns)
            critic_loss = 0.5*tf.math.reduce_mean(advantage**2)

            # print('returns = {}'.format(returns))
            # print('advantage = {}'.format(advantage))
            # print('entropy_loss = {}'.format(entropy_loss))
            # print('actor_loss = {}'.format(actor_loss))
            # print('critic_loss = {}'.format(critic_loss))

            
            
        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)

        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))

        self.actor_loss_metric(actor_loss)
        self.critic_loss_metric(critic_loss)

        # return actor_loss, critic_loss


# from loss_functions import ActorLoss
# class ACER2():
#     def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.memory = deque(maxlen=100000)
#         self.batch_size = 32

#         self.exploration_rate = 1
#         self.exploration_rate_decay = 0.99999975
#         self.exploration_rate_min = 0.1
#         self.gamma = 0.9

#         self.curr_step = 0
#         self.burnin = 1e5  # min. experiences before training
#         self.learn_every = 3   # no. of experiences between updates to Q_online
#         self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

#         self.save_every = 5e5   # no. of experiences between saving Mario Net
#         self.save_dir = save_dir

#         self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

#         self.actor = ActorNet(self.num_actions, 32)
#         self.critic = CriticNet(self.num_actions, 32)

#         self.actor.compile(
#                     optimizer='adam')

#         self.critic.compile(
#                     optimizer='adam')
        
#         self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
#         self.actor_loss = ActorLoss() #tf.math.reduce_mean(-log_probs * advantage) - 0.001*entropy_loss

#     def act(self, state, deterministic=False):
#         logits = self.actor(state)
#         if (deterministic):
#             action = tf.argmax(logits, 1)[0]
#         else:
#             probs = tf.nn.softmax(logits)
#             log_probs = tf.math.log(probs)
#             action = tf.random.categorical(log_probs, 1)[0, 0]
#         return action


#     def cache(self, state, next_state, action, reward, done):
#         """
#         Store the experience to self.memory (replay buffer)

#         Inputs:
#         state (LazyFrame),
#         next_state (LazyFrame),
#         action (int),
#         reward (float),
#         done(bool))
#         """
#         state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
#         next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
#         action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
#         reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
#         done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

#         self.memory.append( (state, next_state, action, reward, done,) )