from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import data
from magenta import music as mm
from magenta.models.music_vae.two_med import configs
from magenta.models.music_vae import TrainedModel
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'run_dir', None,
  'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
  'checkpoint_file', None,
  'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
  'output_dir', '/tmp/music_vae/extracted',
  'The directory where extracted parameters will be saved to.')
flags.DEFINE_string(
  'config', None,
  'The name of the config to use.')
flags.DEFINE_string(
  'input_midi_1', None,
  'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
  'input_midi_2', None,
  'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
  'num_outputs', 5,
  'In `sample` mode, the number of samples to produce. In `interpolate` '
  'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
  'max_batch_size', 8,
  'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
  'temperature', 0.5,
  'The randomness of the decoding process.')
flags.DEFINE_string(
  'log', 'INFO',
  'The threshold for what messages will be logged: '
  'DEBUG, INFO, WARN, ERROR, or FATAL.')
flags.DEFINE_integer(
    'num_data_threads', 4,
    'The number of data preprocessing threads.')
flags.DEFINE_bool(
    'cache_dataset', True,
    'Whether to cache the dataset in memory for improved training speed. May '
    'cause memory errors for very large datasets.')


def _get_input_tensors(dataset, config):
  """Get input tensors from dataset."""
  batch_size = config.hparams.batch_size
  iterator = tf.data.make_one_shot_iterator(dataset)
  (input_sequence, output_sequence, control_sequence,
   sequence_length) = iterator.get_next()
  input_sequence.set_shape(
      [batch_size, None, config.data_converter.input_depth])
  output_sequence.set_shape(
      [batch_size, None, config.data_converter.output_depth])
  if not config.data_converter.control_depth:
    control_sequence = None
  else:
    control_sequence.set_shape(
        [batch_size, None, config.data_converter.control_depth])
  sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

  return {
      'input_sequence': input_sequence,
      'output_sequence': output_sequence,
      'control_sequence': control_sequence,
      'sequence_length': sequence_length
  }

@tf.function
def run(config_map):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

  if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
    raise ValueError(
      'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
  if FLAGS.output_dir is None:
    raise ValueError('`--output_dir` is required.')
  tf.gfile.MakeDirs(FLAGS.output_dir)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config name: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config.data_converter.max_tensors_per_item = None

  logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
      os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
  model = TrainedModel(
    config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
    checkpoint_dir_or_path=checkpoint_dir_or_path)

  # with tf.Graph().as_default():
  #   with tf.device('/device:cpu:0'):
      # Get dataset
  dataset = data.get_dataset(
    config,
    tf_file_reader=tf.data.TFRecordDataset,
    num_threads=FLAGS.num_data_threads,
    is_training=True,
    cache_dataset=FLAGS.cache_dataset)

  tensors = _get_input_tensors(dataset, config)

  everything = model.encode_tensors(tensors["input_sequence"], tensors["sequence_length"])


  # results = model.sample(
  #   n=FLAGS.num_outputs,
  #   length=config.hparams.max_seq_len,
  #   temperature=FLAGS.temperature)
  #
  # basename = os.path.join(
  #   FLAGS.output_dir,
  #   '%s_%s_%s-*-of-%03d.mid' %
  #   (FLAGS.config, FLAGS.mode, date_and_time, FLAGS.num_outputs))
  # logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)
  # for i, ns in enumerate(results):
  #   mm.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

  logging.info('Done.')


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
