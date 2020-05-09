from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from two_med.med_match_model import MedModel

tf.enable_eager_execution()
flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'med_data_1', None,
  'med_data_1')
flags.DEFINE_string(
  'med_data_2', None,
  'med_data_2')
flags.DEFINE_string(
  'log', 'INFO',
  'The threshold for what messages will be logged: '
  'DEBUG, INFO, WARN, ERROR, or FATAL.')


def __clean__(data: np.ndarray):
  indices = np.nonzero(data[:, 0, 0])[0]
  return data[indices, :, :]


# @tf.function
def train_step(model, s, t, loss_object, optimizer, train_loss):
  with tf.GradientTape() as tape:
    predictions = model(s, training=True)
    loss = loss_object(t, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)


# @tf.function
def test_step(model, s, t, loss_object, test_loss):
  predictions = model(s, training=False)
  t_loss = loss_object(t, predictions)

  test_loss(t_loss)


def run():
  upper_med: np.ndarray = __clean__(np.load(FLAGS.med_data_1)) # shape (248, 3, 512)
  bottom_med = __clean__(np.load(FLAGS.med_data_1))

  mean_mu = upper_med[:, 1, :].mean(axis=0)
  np.save('data/mean_mu.npy', mean_mu)
  mean_sigma = upper_med[:, 2, :].mean(axis=0)
  np.save('data/mean_sigma.npy', mean_sigma)

  data_n = upper_med.shape[0]

  for forward in (True, False):
    X, Y = (upper_med, bottom_med) if forward else (bottom_med, upper_med)
    for k in (16, 32, 64, 128):
      for hidden_layer_n in (1, 2):
        model, train_loss, test_loss = train_helper(X, Y, k, hidden_layer_n)
        model.save('data/model_k_{:d}_hln_{:d}_dir_{:s}.h5'.format(k, hidden_layer_n, 'forward' if forward else 'backward'))
        loss_file = 'data/loss_k_{:d}_hln_{:d}_dir_{:s}.npz'.format(k, hidden_layer_n, 'forward' if forward else 'backward')
        np.savez(loss_file, train_loss=train_loss, test_loss=test_loss)
        print('Train loss {:.3f}, test loss {:.3f}'.format(train_loss[-1], test_loss[-1]))


def train_helper(X, Y, k, hidden_layer_n):
  data_n = X.shape[0]
  train_n = int(data_n * 0.9)
  X_train = X[:train_n, ...]
  Y_train = Y[:train_n, ...]
  X_test = X[train_n:, ...]
  Y_test = Y[train_n:, ...]
  train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(32)
  test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(32)
  if hidden_layer_n == 1:
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dropout(0.1, input_shape=(1, 3, 512), dtype=np.float64),
      tf.keras.layers.Dense(k, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(512, activation='relu')
    ])
  else:
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dropout(0.1, input_shape=(1, 3, 512), dtype=np.float64),
      tf.keras.layers.Dense(k, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(k, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(512, activation='relu')
    ])
  loss_object = tf.keras.losses.MeanSquaredError()
  optimizer = tf.keras.optimizers.Adam()
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  EPOCHS = 500
  train_loss_arr = []
  test_loss_arr = []
  for epoch in tqdm(range(EPOCHS)):
    train_loss.reset_states()
    test_loss.reset_states()

    for s, t in train_ds:
      train_step(model, s, t, loss_object, optimizer, train_loss)

    for s, t in test_ds:
      test_step(model, s, t, loss_object, test_loss)

    train_loss_arr.append(train_loss.result())
    test_loss_arr.append(test_loss.result())

    # template = 'Epoch {}, Loss: {}, Test Loss: {}'
    # print(template.format(epoch + 1,
    #                       train_loss.result(),
    #                       test_loss.result()))
  return model, train_loss_arr, test_loss_arr


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run()


def console_entry_point():
  # with tf.device('/device:cpu:0'):
  tf.app.run(main)


# if __name__ == '__main__':
#   console_entry_point()
run()