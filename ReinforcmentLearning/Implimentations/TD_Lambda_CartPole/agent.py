import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from collections import deque

from ReplayBuffers import ExperienceReplay_Cartpole, ExperienceReplay
import time

import numpy as np

from mycustomenvs import CartPoleEnv
from myRLtools import ER_Buffer

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
    self.d3 = layers.Dense(num_hidden_units)
    self.lr3 = layers.LeakyReLU()
    self.a = layers.Dense(num_actions)
    

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    x = self.lr1(x)
    return self.a(x)


class TDLambda():
    def __init__(self, num_actions, num_obs, gamma = 0.99, epsilon=0.90, lam=0.8, batch_size = 128, n_steps=500, num_env=1, replay_buffer_size = 10000, ckpts_num = 100, ckpt_dir = './ac_ckpts'):
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.batch_size = batch_size
        self.num_env=num_env
        self.replay_buffer_size = replay_buffer_size
        #Setting up Experience Replay Buffer
        # self.traj_length = tf.cast(replay_buffer_size/num_env, tf.int64)
        # self.memory = ExperienceReplay_Cartpole(num_env, self.traj_length)
        # self.memory = ExperienceReplay(num_env, self.traj_length)
        self.memory = ER_Buffer(CartPoleEnv(), self.num_env, replay_buffer_size, self.num_obs)

        self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.actor = ActorNet(self.num_actions, 32)
        self.target = ActorNet(self.num_actions, 32)
        self.actor.compile(
                    optimizer='adam')
        self.target.compile(
                    optimizer='adam')
        self.target.set_weights(self.actor.get_weights())
        self.target_update_rate = 100
        self.target_update_count = 0
        

        # Define our metrics
        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        self.total_reward_metric = tf.keras.metrics.Mean('total_reward', dtype=tf.float32)
        
        # Checkpointing
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor_optimizer=self.a_opt, actor_net=self.actor)
        self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=ckpts_num)

        self.actions = [0, 1]
        self.z = None

    def act(self, state, deterministic=False):
        q_est = self.actor(state)
        if (deterministic):
            action = tf.argmax(q_est, axis = 1)
        else:
            if np.random.uniform() < self.epsilon:
                action = tf.argmax(q_est, axis = 1)
            else:
                action = tf.constant(np.random.choice(self.actions, self.num_env))
        return action

    def reset_experience_replay(self):
        print('Resetting Experience Replay Buffer')
        # self.memory.replay_buffer.clear()
        pass
    
    def fill_experience_replay(self, env):
        print('Filling Experience Replay Buffer')
        self.reset_envs(env)
        self.memory.fill(self.actor)
        # self.take_n_steps(env, n_steps = self.traj_length)
    
    def reset_envs(self, env):
        self.state_t1 = env.reset()
        self.z = None


    # @tf.function
    def train(self):

        # Update the weights of the target net
        self.target_update_count += 1
        if(self.target_update_count % self.target_update_rate == 0):
            self.target.set_weights(self.actor.get_weights())

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
        # sample = self.memory.replay_buffer.get_next(sample_batch_size=self.batch_size, num_steps=1)
        state_t1, action_t1, reward_t2, state_t2, done = self.memory.sample(self.batch_size)
        # print('~~~~~~~~~')
        # print(sample)
        
        # state_t1 = tf.squeeze(sample[0][0], axis=1)
        # action_t1 = tf.squeeze(sample[0][1], axis=1)
        # reward_t2 = tf.squeeze(sample[0][2], axis=1)
        # state_t2 = tf.squeeze(sample[0][3], axis=1)
        # done = tf.squeeze(sample[0][4], axis=1)
        total_reward = reward_t2

        # state_t1_mod = tf.concat([state_t1, -state_t1], 0)
        # action_t1_mod = tf.concat([action_t1, 1-action_t1], 0)
        # reward_t2_mod = tf.concat([reward_t2, reward_t2], 0)
        # state_t2_mod = tf.concat([state_t2, -state_t2], 0)
        # done_mod = tf.concat([done, done], 0)
        # total_reward_mod = tf.concat([total_reward, total_reward], 0)
        self.state_t1_mod = tf.cast(state_t1, dtype=tf.float32)
        self.action_t1_mod = tf.cast(tf.expand_dims(action_t1, axis=1), dtype=tf.int32)
        self.reward_t2_mod = tf.cast(tf.expand_dims(reward_t2, axis=1), dtype=tf.float32)
        self.state_t2_mod = tf.cast(state_t2, dtype=tf.float32)
        self.done_mod = tf.cast(tf.expand_dims(done, axis=1), dtype=tf.float32)
        self.total_reward_mod = total_reward

        # print('state_t1 = {}'.format(self.state_t1_mod))
        # print('action_t1 = {}'.format(self.action_t1_mod))
        # print('reward_t2 = {}'.format(self.reward_t2_mod))
        # print('state_t2 = {}'.format(self.state_t2_mod))
        # print('done = {}'.format(self.done_mod))
        
        # mini_batch = list(agent.memory.memory)[0:self.mini_batch_size]
        # mini_batch = Transition(*zip(*mini_batch))
        # state_t1 = tf.reshape(mini_batch[0], shape=(self.mini_batch_size, num_obs))
        # action_t1 = tf.reshape(mini_batch[1], shape=(self.mini_batch_size, 1))
        # state_t2 = tf.reshape(mini_batch[2], shape=(self.mini_batch_size, num_obs))
        # reward_t2 = tf.reshape(mini_batch[3], shape=(self.mini_batch_size, 1))
        # done = tf.reshape(mini_batch[4], shape=(self.mini_batch_size, 1))
        

    
        with tf.GradientTape() as tape:
            self.q_s1_est = self.actor(self.state_t1_mod, training=True)
            self.q_s2_est = self.target(self.state_t2_mod, training=False)
            
            # self.q_s1_gathered = tf.gather(q_s1_est, action_t1_mod, axis=1, batch_dims=1)
            self.q_s1_gathered = tf.gather(self.q_s1_est, self.action_t1_mod, axis=1, batch_dims=1)
            self.q_s2_max = tf.expand_dims(tf.reduce_max(self.q_s2_est, axis=1), axis=1)

            self.returns = (self.reward_t2_mod + self.gamma*self.q_s2_max)*(1-self.done_mod)
            self.delta = self.returns - self.q_s1_gathered
            self.loss = tf.math.reduce_sum(self.delta**2)
            
            
            
        self.grads = tape.gradient(self.loss, self.actor.trainable_variables)
        # if(self.z is None):
        #     self.z = self.grads
        # else:
        #     for g in range(len(self.grads)):
        #         self.z[g] *= self.gamma*self.lam
        #         self.z[g] += self.grads[g]

        
        # self.a_opt.apply_gradients(zip(self.z, self.actor.trainable_variables))
        self.a_opt.apply_gradients(zip(self.grads, self.actor.trainable_variables))

        self.actor_loss_metric(self.loss)

        self.num_dones = tf.math.reduce_sum(tf.cast(self.done_mod, dtype=tf.float32) )

        if(self.num_dones > 0):
            # print('num_dones = {}'.format(num_dones))
            # print('total_reward_mod = {}'.format(total_reward_mod))
            # print((total_reward_mod*done_mod)/num_dones)
            # avg_batch_total_rewards = tf.math.reduce_sum(total_reward_mod*done_mod)/num_dones
            # print('avg_batch_total_rewards = {}'.format(avg_batch_total_rewards))
            self.total_reward_metric(self.batch_size/self.num_dones)
            

        # return actor_loss, critic_loss

    def load_checkpoint(self, path=None):
        if(path is None):
            #Try and load the latest checkpoint if it exists
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")
        try:
            #Try and load the specified checkpoint if it exists
            self.ckpt.restore(path)
            print("Restored from path {}".format(path))
        except:
            print("Checkpoint {} not found".format(path))
            print("Initializing from scratch.")
            


    def train_and_checkpoint(self, save_freq = 100):
        self.ckpt.step.assign_add(1)
        if int(self.ckpt.step) % save_freq == 0:
            save_path = self.manager.save()
            # print("Saved AC checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        
        self.train()

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