# Read all elements in the replay buffer:
print('~~~~~~~~~')
print('~~~~~~~~~')
trajectories = agent.memory.replay_buffer.gather_all()
print("Trajectories from gather all:")
print(tf.nest.map_structure(lambda t: t.shape, trajectories))
# print(trajectories)
print(trajectories[0])
# print(trajectories[0][:,0,:]) #First index is the data type, second is [entry in batch, batch number, data number]
# print(trajectories[3][:,0,:]) #First index is the data type, second is [entry in batch, batch number, data number]

#%%
print()
print('Trajectories state_t1 = \n{}'.format(trajectories[0]))
#%%
for _ in range(1):
    # Get one sample from the replay buffer with batch size (self.batch_size) and 1 timestep:
    sample = agent.memory.replay_buffer.get_next(sample_batch_size=2, num_steps=2)
    
    state_t1 = sample[0][0][:,0] #First State
    action_t1 = sample[0][1][:,0] #First Action
    state_tn = sample[0][3][:,-1] #last State

    # Calculate Window Returns
    tmp_rewards = sample[0][2]
    tmp_dones = sample[0][4]
    # tmp_rewards = tf.constant([[[1.],[2.]],[[3.],[4.]]])
    reward_tn = tf.zeros(shape=tmp_rewards[0].shape)
    sum_of_dones = tf.zeros(shape=tmp_dones[0].shape)
    num_steps = len(tmp_rewards[0])
    gamma = 0.9
    for i in range(num_steps):
        sum_of_dones += tf.clip_by_value(tmp_dones[:,i], 0, 1)
        reward_tn += (gamma**i)*tmp_rewards[:,i]*(1-sum_of_dones)
        print('sum_of_dones')
        print(sum_of_dones)
    print('reward_tn')  
    print(reward_tn)
    
    
    
    # total_reward = tf.squeeze(sample[0][5], axis=1)
    # print()
    # print('sampled state_t1 = \n{}'.format(state_t1))

#%%
tmp_dones = sample[0][4]
tmp_dones = tf.constant([[[1],[0]],[[0],[1]]])
print('tmp_dones')
print(tmp_dones)

# tmp_dones_index = -tf.ones(shape=tmp_dones[0].shape)
tmp_dones_index = tf.ones(shape=tmp_dones[0].shape)*(num_steps-1)
tmp_dones_index = tf.cast(tmp_dones_index, dtype=tf.int32)

print('tmp_dones_index')
print(tmp_dones_index)
# state_tn = tf.gather(sample[0][3]) #last State
state_tn = tf.gather(sample[0][3], tmp_dones_index, axis=1, batch_dims=1)
print('state_tn')
print(state_tn)

#%%

test = tf.sequence_mask([5, 0, 3, 2], 5, dtype=tf.int32)
print(test)
#%%
where_mask = tf.where(agent.dones_sample==1)
print(where_mask)

index_holder = tf.ones(shape=(agent.batch_size, agent.n_steps), dtype=tf.int64)*5
index_holder = tf.where(tf.squeeze(agent.dones_sample)==1, 0, index_holder)
print(index_holder)
# test = tf.sequence_mask(index_holder, 5, dtype=tf.int32)
# print(test)

#%%
tf.random.set_seed(31)

n = 4 # td_n
m = 6 # sequences in batch
gamma = 0.9

gamma_vec = tf.cast(tf.reshape(gamma**tf.linspace(0,n-1,n),shape = [-1,1]),'float32' )

sample_batch_rewards = 0.5*tf.ones([m,n])
sample_dones = tf.cast(tf.random.uniform([m,n])>0.8,'int64')

idx = tf.argmax(sample_dones,axis = 1)+1 + (n-1)*(tf.cast(tf.reduce_sum(sample_dones,axis = 1)==0,'int64'))
mask = tf.cast(tf.sequence_mask(idx,n),'float32')
masked_rewards = sample_batch_rewards*mask

R = tf.matmul(masked_rewards,gamma_vec)

print(f'sample_batch_rewards:\n {sample_batch_rewards}')
print(f'sample_dones:\n {sample_dones}')
print(f'idx:\n {idx}')
print(f'mask:\n {mask}')
print(f'masked_rewards:\n {masked_rewards}')
print(f'R:\n {R}')




    
    