from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
from os import listdir
from os.path import isfile, join

import tensorflow.compat.v1 as tf
from tqdm import tqdm

from magenta import music as mm
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import data
from magenta.models.music_vae.two_med import configs

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
  'output_npy', '/tmp/music_vae/extracted',
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
flags.DEFINE_string(
  'midi_dir', '',
  'Dir storing the original midi files.')


# def _check_extract_examples(input_ns, path, input_number):
#   """Make sure each input returns exactly one example from the converter."""
#   tensors = config.data_converter.to_tensors(input_ns).outputs
#   if not tensors:
#     print(
#       'MusicVAE configs have very specific input requirements. Could not '
#       'extract any valid inputs from `%s`. Try another MIDI file.' % path)
#     sys.exit()
#   elif len(tensors) > 1:
#     basename = os.path.join(
#       FLAGS.output_dir,
#       '%s_input%d-extractions_%s-*-of-%03d.mid' %
#       (FLAGS.config, input_number, date_and_time, len(tensors)))
#     for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
#       mm.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))
#     print(
#       '%d valid inputs extracted from `%s`. Outputting these potential '
#       'inputs as `%s`. Call script again with one of these instead.' %
#       (len(tensors), path, basename))
#     sys.exit()


# @tf.function
def run(config_map):
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

  if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
    raise ValueError(
      'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
  if FLAGS.output_npy is None:
    raise ValueError('`--output_npy` is required.')
  # tf.gfile.MakeDirs(FLAGS.output_npy)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config name: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config.data_converter.max_tensors_per_item = None

  # Check midi file
  if not os.path.exists(FLAGS.midi_dir):
    raise ValueError('MIDI dir not found: %s' % FLAGS.midi_dir)



  logging.info(
    'Attempting to extract examples from input MIDIs using config `%s`...',
    FLAGS.config)

  logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
      os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
  model = TrainedModel(
    config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
    checkpoint_dir_or_path=checkpoint_dir_or_path)

  logging.info('Extracting latent parameters...')
  midi_files = [f for f in listdir(FLAGS.midi_dir) if isfile(join(FLAGS.midi_dir, f))]
  extraction = np.zeros((len(midi_files), 3, 512))
  for i, _midi_file in tqdm(enumerate(midi_files)):
    midi_file = FLAGS.midi_dir + '/' + _midi_file
    try:
      input_midi = mm.midi_file_to_note_sequence(midi_file)
    except:
      continue
    tensor = config.data_converter.to_tensors(input_midi).outputs
    if not tensor:
      # logging.info('Skipping {:s}'.format(_midi_file))
      continue
    z, mu, sigma = model.encode([input_midi])
    extraction[i, 0, :] = z
    extraction[i, 1, :] = mu
    extraction[i, 2, :] = sigma

  np.save(FLAGS.output_npy, extraction)

  logging.info('Done.')


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  # tf.disable_v2_behavior()
  with tf.device('/device:cpu:0'):
    tf.app.run(main)


if __name__ == '__main__':
  # midi_dir = '/home/xdroid/Shared/dataset/test-2'
  # converter = data.OneHotMelodyConverter(
  #   skip_polyphony=False,
  #   max_tensors_per_notesequence=50,
  #   max_bars=100,  # Truncate long melodies before slicing.
  #   slice_bars=2,
  #   steps_per_quarter=4)
  # midi_files = [f for f in listdir(midi_dir) if isfile(join(midi_dir, f))]
  # for _midi_file in midi_files:
  #   midi_file = join(midi_dir, _midi_file)
  #   input_midi = mm.midi_file_to_note_sequence(midi_file)
  #   tensor = converter.to_tensors(input_midi).outputs
  # sys.exit()
  console_entry_point()
