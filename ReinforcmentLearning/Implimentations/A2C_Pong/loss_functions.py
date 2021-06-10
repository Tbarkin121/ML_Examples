import tensorflow as tf

class ActorLoss():
  def __call__(self, log_probs, advantage, entropy_loss):
    return  tf.math.reduce_mean(-log_probs * advantage) - 0.001*entropy_loss


