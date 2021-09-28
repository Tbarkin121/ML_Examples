import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

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
    # self.a = layers.Dense(num_actions, activation='tanh')
    # self.a = layers.Dense(num_actions, activation='sigmoid')
    self.a = layers.Dense(num_actions)
    

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    # x = tf.keras.activations.tanh(x)
    x = self.lr1(x)
    x = layers.BatchNormalization()(x)
    x = self.d2(x)
    x = self.lr2(x)
    x = layers.BatchNormalization()(x)
    x = self.d3(x)
    x = self.lr3(x)
    x = layers.BatchNormalization()(x)
    return self.a(x)

test = ActorNet(4, 16)
test_data = tf.random.uniform(shape=(1,4), minval=-1, maxval=1)
test_output = test(test_data)

print(test_data)
print(test_output)

#%%
import tensorflow_probability as tfp

p = [[.1, .2, .7], [.3, .3, .4]]  # Shape [2, 3]
dist = tfp.distributions.Multinomial(total_count=[4., 5], probs=p)


# print(dist.sample(1))

counts = [[2., 1, 1], [3, 1, 1]]
print(dist.prob(counts))  # Shape [2]
    
print(dist.sample(1))

# p = [.2, .3, .5]
# dist = Multinomial(total_count=4., probs=p)

#%%

p = [.2, .3, .5]
dist = tfp.distributions.Multinomial(total_count=4., probs=p)
print(dist.prob(p))  # Shape []

# counts same shape as p.
counts = [1., 0, 3]
print(dist.prob(counts))  # Shape []

# p will be broadcast to [[.2, .3, .5], [.2, .3, .5]] to match counts.
counts = [[1., 2, 1], [2, 2, 0]]
print(dist.prob(counts))  # Shape [2]


# # p will be broadcast to shape [5, 7, 3] to match counts.
# counts = [[...]]  # Shape [5, 7, 3]
# print(dist.prob(counts))  # Shape [5, 7]