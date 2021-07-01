import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from ExperienceReplay import ER_Buffer
from collections import deque

from ReplayBuffers import ExperienceReplay_Cartpole, ExperienceReplay
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
    self.d3 = layers.Dense(num_hidden_units)
    self.lr3 = layers.LeakyReLU()
    self.a = layers.Dense(num_actions)
    

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    x = self.lr1(x)
    return self.a(x)


class TDLambda():
    def __init__(self, num_actions, num_obs, gamma = 0.99, epsilon=0.90, lam=0.8, n_steps=1, batch_size = 128, num_env=1, replay_buffer_size = 10000, ckpts_num = 100, ckpt_dir = './ac_ckpts'):
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_env=num_env

        #Setting up Experience Replay Buffer
        self.traj_length = tf.cast(replay_buffer_size/num_env, tf.int64)
        self.memory = ExperienceReplay_Cartpole(num_env, self.traj_length)
        # self.memory = ExperienceReplay(num_env, self.traj_length)

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

        self.gamma_exp = tf.expand_dims(tf.cast(tf.range(self.n_steps), dtype=tf.float32), axis=1)
        self.gamma_vec = self.gamma**self.gamma_exp

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
        self.memory.replay_buffer.clear()
        # pass
    
    def fill_experience_replay(self, env):
        print('Filling Experience Replay Buffer')
        self.reset_envs(env)
        self.take_n_steps(env, n_steps = self.traj_length)
    
    def reset_envs(self, env):
        self.state_t1 = env.reset()
        self.z = None

    # def env_step(action: np.ndarray, env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #   """Returns state, reward and done flag given an action."""

    #   state, reward, done, _ = env.step(action)
    #   return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

    # def tf_env_step(action: tf.Tensor, env) -> List[tf.Tensor]:
    #   return tf.numpy_function(env_step, [action, env], [tf.float32, tf.float32, tf.int32])

    def take_n_steps(self, env, n_steps=1):
        # print('Filling Experience Replay Buffer')
        # print(env.num_envs)
        
        for _ in range(n_steps):
            # action_t1_test = np.zeros(self.num_env, dtype=int)
            self.action_t1 = self.act(self.state_t1, deterministic=False)
            env.step_async(self.action_t1.numpy())
            self.state_t2, self.reward_t2, self.done, total_reward_dict = env.step_wait()
            self.action_t1 = tf.cast(tf.reshape(self.action_t1, shape=(self.num_env,1)),tf.int32)
            self.reward_t2 = tf.reshape(self.reward_t2, shape=(self.num_env,1))
            self.done = tf.cast(tf.reshape(self.done, shape=(self.num_env,1)), dtype=tf.float32)

            self.total_reward = [item['total_rewards'] for item in total_reward_dict]
            self.total_reward = tf.reshape(self.total_reward, shape=(self.num_env,1))

            values = (self.state_t1, self.action_t1, self.reward_t2, self.state_t2, self.done, self.total_reward)
            values_batched = tf.nest.map_structure(lambda t: tf.stack(t), values)
            self.memory.replay_buffer.add_batch(values_batched)
            self.state_t1 = self.state_t2

    @tf.function
    def fucky_forloop(self):
        self.done_mask = np.ones(shape=self.rewards_sample.shape)
        for loc in self.done_loc:
            self.done_mask[loc[0], loc[1]:] = 0

    def funky_fauxloop(self):
        idx = tf.argmax(self.dones_sample,axis = 1)+1 + (self.n_steps-1)*(tf.cast(tf.reduce_sum(self.dones_sample,axis = 1)==0,'int64'))
        self.done_mask = tf.cast(tf.sequence_mask(idx, self.n_steps),'float32')
        
    def no_dones(self): 
        self.done_mask = np.ones(shape=self.rewards_sample.shape)

    def states_n_stuff(self):
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

        sample = self.memory.replay_buffer.get_next(sample_batch_size=self.batch_size, num_steps=self.n_steps)
        # print('~~~~~~~~~')
        # print(sample)
        
        self.state_t1_sample = sample[0][0][:,0] #First State
        self.action_t1_sample = sample[0][1][:,0] #First Action
        self.rewards_sample = sample[0][2]
        self.state_tn_sample = sample[0][3][:,-1] #last State
        self.dones_sample = sample[0][4]


        
        self.done_loc = tf.where(self.dones_sample==1)

        # self.fucky_forloop()
        self.funky_fauxloop()
        # self.no_dones()

        self.done_mask2 = np.ones(shape=(self.batch_size, 1))
        if(len(self.done_loc)>0):
            self.done_mask2[self.done_loc[:,0].numpy()]=0
        
        # This one is for funky_forloop and no_dones
        # self.returns = tf.matmul(tf.squeeze(self.rewards_sample*self.done_mask, axis=2), self.gamma_vec)
        
        # Because of a slightly different shape I haven't fixed yet, this returns is used for funky_fauxloop
        self.returns = tf.matmul(tf.squeeze(self.rewards_sample, axis=2)*tf.squeeze(self.done_mask, axis=1), self.gamma_vec)

    def gradient_stuff(self):
        with tf.GradientTape() as tape:
            self.q_s1_est = self.actor(self.state_t1_sample, training=True)
            self.q_sn_est = self.target(self.state_tn_sample, training=False)
            
            self.q_s1_gathered = tf.gather(self.q_s1_est, self.action_t1_sample, axis=1, batch_dims=1)
            self.q_sn_max = tf.expand_dims(tf.reduce_max(self.q_sn_est, axis=1), axis=1)
            
            self.returns += self.gamma**self.n_steps*self.q_sn_max*self.done_mask2
            self.delta = self.returns - self.q_s1_gathered
            self.loss = tf.math.reduce_mean(self.delta**2)
            
            
        self.grads = tape.gradient(self.loss, self.actor.trainable_variables)
        
        self.a_opt.apply_gradients(zip(self.grads, self.actor.trainable_variables))

    # @tf.function
    def train(self):

        # Update the weights of the target net
        self.target_update_count += 1
        if(self.target_update_count % self.target_update_rate == 0):
            self.target.set_weights(self.actor.get_weights())

        
        self.states_n_stuff()

        self.gradient_stuff()

        self.actor_loss_metric(self.loss)

        num_dones = tf.math.reduce_sum(tf.cast(self.dones_sample, dtype=tf.float32) )

        if(num_dones > 0):
            # print('num_dones = {}'.format(num_dones))
            # print('total_reward_mod = {}'.format(total_reward_mod))
            # print((total_reward_mod*done_mod)/num_dones)
            # avg_batch_total_rewards = tf.math.reduce_sum(total_reward_mod*done_mod)/num_dones
            # print('avg_batch_total_rewards = {}'.format(avg_batch_total_rewards))
            self.total_reward_metric(self.n_steps*self.batch_size/num_dones)
            

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