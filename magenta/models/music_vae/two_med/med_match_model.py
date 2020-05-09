import tensorflow as tf
from tensorflow.keras import Model


class MedModel(Model):
  def __init__(self, k, t):
    super(MedModel, self).__init__()
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(k, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(t, activation='relu')
    ])

  def call(self, x, *args, **kwargs):
    return self.model(x, *args, **kwargs)