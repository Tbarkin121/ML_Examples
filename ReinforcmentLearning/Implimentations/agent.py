import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from ExperienceReplay import ER_Buffer
from collections import deque

from ReplayBuffers import ExperienceReplay_Cartpole
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
    x = self.d3(x)
    x = self.lr3(x)
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
    self.d3 = layers.Dense(num_hidden_units)
    self.lr3 = layers.LeakyReLU()
    # self.critic = layers.Dense(num_actions)
    self.c = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    # x = tf.keras.activations.tanh(x)
    x = self.lr1(x)
    x = self.d2(x)
    x = self.lr2(x)
    x = self.d3(x)
    x = self.lr3(x)
    return self.c(x)

class ACER():
    def __init__(self, num_actions, num_obs, gamma = 0.99, batch_size = 128, n_steps=500, num_env=1, replay_buffer_size = 10000, ckpts_num = 100, ckpt_dir = './ac_ckpts'):
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_env=num_env

        #Setting up Experience Replay Buffer
        self.traj_length = tf.cast(replay_buffer_size/num_env, tf.int64)
        self.memory = ExperienceReplay_Cartpole(num_env, self.traj_length)

        self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.actor = ActorNet(self.num_actions, 32)
        self.critic = CriticNet(self.num_actions, 32)

        self.actor.compile(
                    optimizer='adam')

        self.critic.compile(
                    optimizer='adam')
        
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.I = 1
        # Define our metrics
        self.actor_loss_metric = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        self.critic_loss_metric = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
        self.total_reward_metric = tf.keras.metrics.Mean('total_reward', dtype=tf.float32)
        self.logits1_metric = tf.keras.metrics.Mean('logits1', dtype=tf.float32)
        self.logits2_metric = tf.keras.metrics.Mean('logits2', dtype=tf.float32)
        self.probs1_metric = tf.keras.metrics.Mean('probs1', dtype=tf.float32)
        self.probs2_metric = tf.keras.metrics.Mean('probs2', dtype=tf.float32)
        
        # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

        # Checkpointing
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor_optimizer=self.a_opt, actor_net=self.actor, critic_optimizer=self.c_opt, critic_net=self.critic)
        self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=ckpts_num)

          
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
        total_reward = tf.squeeze(sample[0][5], axis=1)

        state_t1_mod = tf.concat([state_t1, -state_t1], 0)
        action_t1_mod = tf.concat([action_t1, 1-action_t1], 0)
        reward_t2_mod = tf.concat([reward_t2, reward_t2], 0)
        state_t2_mod = tf.concat([state_t2, -state_t2], 0)
        done_mod = tf.concat([done, done], 0)
        total_reward_mod = tf.concat([total_reward, total_reward], 0)
        # state_t1_mod = state_t1
        # action_t1_mod = action_t1
        # reward_t2_mod = reward_t2
        # state_t2_mod = state_t2
        # done_mod = done
        # total_reward_mod = total_reward

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
            logits = self.actor(state_t1_mod, training=True)
            values_t1 = self.critic(state_t1_mod, training=True)
            values_t2 = self.critic(state_t2_mod, training=True)
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
            
            probs_actions = tf.gather(probs, action_t1_mod, axis=1, batch_dims=1)
            log_probs = tf.math.log(probs_actions)
            # print('action_t1 = {}'.format(action_t1))
            # print('probs_actions = {}'.format(probs_actions))
            # print('probs_actions.shape = {}'.format(probs_actions.shape))
            # print('log_probs = {}'.format(log_probs))
            # print('log_probs.shape = {}'.format(log_probs.shape))
            
            
            returns = (reward_t2_mod + self.gamma*values_t2)*(1-done_mod)
            # returns = reward_t2_mod + (self.gamma*values_t2)*(1-done_mod)
            # returns = tf.cast(reward_t2_mod, 'float32') + self.gamma*values_t2
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
        num_dones = tf.math.reduce_sum(tf.cast(done_mod, dtype=tf.float32) )

        if(num_dones > 0):
            # print('num_dones = {}'.format(num_dones))
            # print('total_reward_mod = {}'.format(total_reward_mod))
            # print((total_reward_mod*done_mod)/num_dones)
            avg_batch_total_rewards = tf.math.reduce_sum(total_reward_mod*done_mod)/num_dones
            # print('avg_batch_total_rewards = {}'.format(avg_batch_total_rewards))
            self.total_reward_metric(avg_batch_total_rewards)
            

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