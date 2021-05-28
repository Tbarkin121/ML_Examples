import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
from ExperienceReplay import ExperienceReplay

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
    def __init__(self, env, num_actions, gamma = 0.99, mem_size = 100000, batch_size = 1024, mini_batch_size = 128, n_mini_batches=1):
        self.env = env
        self.num_actions = num_actions
        self.gamma = gamma
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_mini_batches = n_mini_batches
        self.memory = ExperienceReplay(self.mem_size)

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
            action = tf.random.categorical(log_probs, 1)[0, 0]
        return action
    def reset_replay_buffer(self):
        # Reset memory buffer
        self.memory = ExperienceReplay(self.mem_size)
    
    def fill_replay_buff(self, n_samples):        
        # Reset Environment
        state = tf.constant(env.reset(), dtype=tf.float32)
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)
        for i in range(n_samples):
          # Run the model and to get action probabilities and critic value
          action = self.act(state)

          # print('action = {}'.format(action))
          # Apply action to the environment to get next state and reward
          next_state, reward, done = tf_env_step(action)
          next_state = tf.expand_dims(next_state, 0)
        
          # print('next_state = {}'.format(next_state))
          # print('reward = {}'.format(reward))
          # print('done = {}'.format(done))
          # Store the transition in memory
          # print('')
          #If Episode is done, reset the environment
          if tf.cast(done, tf.bool):
            self.memory.push(state, action, next_state, 0, done)
            state = tf.constant(env.reset(), dtype=tf.float32)
            state = tf.expand_dims(state, 0)
          else:
            self.memory.push(state, action, next_state, reward, done)
            state=next_state

    # @tf.function
    def train(self, n_mini_batches = 1):
        self.fill_replay_buff(self.batch_size)
        for _ in range(n_mini_batches):
            mini_batch = self.memory.sample(self.mini_batch_size)
            # mini_batch = list(agent.memory.memory)[0:self.mini_batch_size]
            mini_batch = Transition(*zip(*mini_batch))
            state_t1 = tf.reshape(mini_batch[0], shape=(self.mini_batch_size, num_obs))
            action_t1 = tf.reshape(mini_batch[1], shape=(self.mini_batch_size, 1))
            state_t2 = tf.reshape(mini_batch[2], shape=(self.mini_batch_size, num_obs))
            reward_t2 = tf.reshape(mini_batch[3], shape=(self.mini_batch_size, 1))
            done = tf.reshape(mini_batch[4], shape=(self.mini_batch_size, 1))
            
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                logits = self.actor(state_t1, training=True)
                values_t1 = self.critic(state_t1, training=True)
                values_t2 = self.critic(state_t2, training=True)
                # print('logits = {}'.format(logits))
                # print('action_t1 = {}'.format(action_t1))
                
                # print('logit_actions = {}'.format(logit_actions))
                # print('logit_actions.shape = {}'.format(logit_actions.shape))
                # time.sleep(5)

                probs = tf.nn.softmax(logits)
                l = logits.numpy()
                p = probs.numpy()
                self.logits1_metric(l[0])
                self.logits2_metric(l[1])
                self.probs1_metric(p[0])
                self.probs2_metric(p[1])

                probs_actions = tf.gather(probs, action_t1, axis=1, batch_dims=1)
                log_probs = tf.math.log(probs_actions)
                # print('probs_actions = {}'.format(probs_actions))
                # print('probs_actions.shape = {}'.format(probs_actions.shape))
                # print('log_probs = {}'.format(log_probs))
                # print('log_probs.shape = {}'.format(log_probs.shape))
                # time.sleep(5)

                returns = (tf.cast(reward_t2, 'float32')) + self.gamma*values_t2*(1-tf.cast(done, 'float32'))
                # returns = -tf.cast(done, 'float32') + self.gamma*values_t2*(1-tf.cast(done, 'float32'))
                advantage =  returns - values_t1 

                entropy_loss = -tf.math.reduce_mean(probs_actions*log_probs)
                # actor_loss = -log_probs*advantage
                # actor_loss = tf.math.reduce_mean(-self.I*log_probs * advantage) - 0.00001*entropy_loss
                actor_loss = tf.math.reduce_mean(-log_probs * advantage) - 0.001*entropy_loss
                self.I *= self.gamma
                critic_loss = 0.5*self.huber_loss(values_t1, returns)
                # critic_loss = 0.5*tf.math.reduce_mean(advantage**2)

                
                
            grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
            grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)

            self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
            self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))

            self.actor_loss_metric(actor_loss)
            self.critic_loss_metric(critic_loss)

        # return actor_loss, critic_loss
    def summary(self):
        pass
        # state = tf.constant(env.reset(), dtype=tf.float32)
        # state = tf.expand_dims(state, 0)
        # self.actor.build(input_shape=(1,4))
        # self.actor.summary()
        # self.critic.build(input_shape=(1,4))
        # self.critic.summary()