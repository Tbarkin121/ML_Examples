import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import numpy as np

#%%
class ExperienceReplay_Cartpole():
    def __init__(self, batch_size, max_length):
        self.batch_size = batch_size
        self.max_length = max_length

        self.data_spec =  (
            tf.TensorSpec([4], tf.float32, 'state_t1'),
            tf.TensorSpec([1], tf.int32, 'action_t1'),
            tf.TensorSpec([1], tf.float32, 'reward_t2'),
            tf.TensorSpec([4], tf.float32, 'state_t2'),
            tf.TensorSpec([1], tf.float32, 'done'),
            tf.TensorSpec([1], tf.float32, 'total_reward')
        )

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        self.data_spec,
                        batch_size=self.batch_size,
                        max_length=self.max_length)

#%%
class ExperienceReplay():
    def __init__(self, batch_size, max_length):
        self.batch_size = batch_size
        self.max_length = max_length

        self.data_spec =  (
            tf.TensorSpec([6], tf.float32, 'state_t1'),
            tf.TensorSpec([1], tf.int32, 'action_t1'),
            tf.TensorSpec([1], tf.float32, 'reward_t2'),
            tf.TensorSpec([6], tf.float32, 'state_t2'),
            tf.TensorSpec([1], tf.bool, 'done'),
            tf.TensorSpec([1], tf.float32, 'total_reward')
        )

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        self.data_spec,
                        batch_size=self.batch_size,
                        max_length=self.max_length)  

# ER = ExperienceReplay(5,100)
# state_t1 = tf.constant(
#     1 * np.ones(ER.data_spec[0].shape.as_list(), dtype=np.float32))
# action_t1 = tf.constant(
#     2 * np.ones(ER.data_spec[1].shape.as_list(), dtype=np.float32))
# reward_t2 = tf.constant(
#     3 * np.ones(ER.data_spec[2].shape.as_list(), dtype=np.float32))
# state_t2 = tf.constant(
#     4 * np.ones(ER.data_spec[3].shape.as_list(), dtype=np.float32))
# done = tf.constant(
#     5 * np.ones(ER.data_spec[4].shape.as_list(), dtype=np.float32))
# values = (state_t1, action_t1, reward_t2, state_t2, done)
# values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * ER.batch_size), values)

# print('values')
# print()
# print(values)
# print()

# print('values_batched')
# print()
# print(values_batched)
# print()

# multiplier = 1
# for _ in range(3):
#     state_t1 = tf.constant(
#         1 * multiplier * np.ones(data_spec[0].shape.as_list(), dtype=np.float32))
#     action_t1 = tf.constant(
#         2 * multiplier * np.ones(data_spec[1].shape.as_list(), dtype=np.float32))
#     reward_t2 = tf.constant(
#         3 * multiplier * np.ones(data_spec[2].shape.as_list(), dtype=np.float32))
#     state_t2 = tf.constant(
#         4 * multiplier * np.ones(data_spec[3].shape.as_list(), dtype=np.float32))
#     done = tf.constant(
#         5 * multiplier * np.ones(data_spec[4].shape.as_list(), dtype=np.float32))

#     values = (state_t1, action_t1, reward_t2, state_t2, done)
#     values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size), values)
#     replay_buffer.add_batch(values_batched)
#     multiplier += 1

# #print('values')
# #print(values)

# # Read all elements in the replay buffer:
# trajectories = replay_buffer.gather_all()
# print("Trajectories from gather all:")
# print(tf.nest.map_structure(lambda t: t.shape, trajectories))
# print(trajectories[0][:,0,:]) #First index is the data type, second is [entry in batch, batch number, data number]


# # Get one sample from the replay buffer with batch size 10 and 1 timestep:
# print('sample')
# sample = replay_buffer.get_next(sample_batch_size=4, num_steps=1)
# print(sample)
# print()
# print()
# print(sample[0][2][:,0,0])