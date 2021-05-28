from tensorflow.keras.losses import Loss
class ActorLoss(Loss):
  def call(self, y_true, y_pred):
    return tf.reduce_mean((y_pred - y_true)**2, axis=-1)

testActorLoss = ActorLoss()
y_true = tf.Variable([1,2], dtype='float32')
y_pred = tf.Variable([3,3], dtype='float32')
print(y_true.shape)
loss = testActorLoss(y_true, y_pred)
print('loss = {}'.format(loss))

