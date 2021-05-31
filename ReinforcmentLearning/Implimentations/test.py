# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:46:20 2021

@author: tylerbarkin
"""

from loss_functions import ActorLoss
import tensorflow as tf

testActorLoss = ActorLoss()
log_probs = tf.Variable([1,2], dtype='float32')
advantage = tf.Variable([3,3], dtype='float32')
entropy_loss = tf.Variable([3,3], dtype='float32')

print(log_probs.shape)
loss = testActorLoss(log_probs, advantage, entropy_loss)
print('loss = {}'.format(loss))