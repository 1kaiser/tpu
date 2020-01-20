# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests that the resnet model loads without error."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from official.resnet import resnet_model
import resnet_preprocessing

from tensorflow.python import pywrap_tensorflow
import os
import time

from tensorflow.core.protobuf import config_pb2
import tqdm

filename = 'gs://cloud-tpu-test-datasets/fake_imagenet/train-00000-of-01024'
filename = 'gs://gpt-2-poetry/data/imagenet/out/train-00000-of-01024'
filename = 'gs://gpt-2-poetry/data/imagenet/out/validation-00000-of-00128'

params = {}
params['label_smoothing'] = 0.1
params['num_label_classes'] = 1001
params['beta1'] = 0.9
params['beta2'] = 0.999
params['epsilon'] = 1e-9
#params['lr'] = 0.245
params['lr'] = float(os.environ['LR']) if 'LR' in os.environ else 0.000055
params['batch_size'] = int(os.environ['BATCH_SIZE']) if 'BATCH_SIZE' in os.environ else 16
params['num_cores'] = 1
params['image_size'] = 224
#params['prefetch_mb'] = 2048,  # Amount of data to prefetch (megabytes), 0 = disable prefetching.
params['prefetch_mb'] = 128  # Amount of data to prefetch (megabytes), 0 = disable prefetching.
params['buffer_mb'] = 16  # Read buffer size (megabytes).
params['repeat'] = bool(os.environ['REPEAT']) if 'REPEAT' in os.environ else False
params['train_iterations'] = int(os.environ['TRAIN_ITERATIONS']) if 'TRAIN_ITERATIONS' in os.environ else 4

def iterate_imagenet(sess):
  image_preprocessing_fn = resnet_preprocessing.preprocess_image
  dset = tf.data.TFRecordDataset(filename, buffer_size=(params['buffer_mb']<<20))
  keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, ''), 'image/class/label': tf.FixedLenFeature([], tf.int64, -1), }
  def parse(value):
    return tf.parse_single_example(value, keys_to_features)
  dset = dset.map(parse)
  tfr_shape = [params['batch_size'], params['image_size'], params['image_size'], 3]
  bytes_per_item = np.prod(tfr_shape) * np.dtype(np.float32).itemsize
  #if shuffle_mb > 0:
    #dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
  if params['repeat']:
    dset = dset.repeat()
  if params['prefetch_mb'] > 0:
    dset = dset.prefetch(((params['prefetch_mb'] << 20) - 1) // bytes_per_item + 1)
  dset = dset.batch(params['batch_size'])
  tf_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
  init = tf_iterator.make_initializer(dset)
  sess.run(init)
  tf_iterator_next = tf_iterator.get_next()
  def get_next(sess):
    parsed = sess.run(tf_iterator_next)
    def body():
      for label, image_encoded in zip(parsed['image/class/label'], parsed['image/encoded']):
        image_bytes = tf.reshape(image_encoded, shape=[])
        image = image_preprocessing_fn(image_bytes=image_bytes, is_training=True, image_size=224, use_bfloat16=False)
        label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
        if True:
          yield label, image
        else:
          image_result = sess.run(image)
          label_result = sess.run(label)
          yield label_result, image_result
    results = list(body())
    labels = [x[0] for x in results]
    images = [x[1] for x in results]
    return labels, images
  return get_next

class Namespace(object):
  pass

state = Namespace()

#state.split_param_count = 1e4
state.split_param_count = 2e6

def split_by_params(vs, n=None, f=None):
  if n is None:
    n = state.split_param_count
  if f is None:
    f = lambda x: np.prod(x.shape.as_list())
  i = 0
  xs = []
  for variable in vs:
    xs.append(variable)
    count = f(variable)
    i += count
    if i >= n:
      yield xs
      xs = []
      i = 0
  yield xs

def assign_values(variables, values, session=None, timeout_in_ms=6000000):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)]) # TODO: bfloat16 support
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  session.run(ops, vals, options=options)

import re

def variable_name(variable):
  if re.match(r'core[0-9]+/', variable.name):
    return variable.name.split('/', 1)[-1]
  return variable.name

def grab_values(variables, reader, reshape=False, scope=''):
  for variable in variables:
    name = os.path.join(scope, variable_name(variable).split(':')[0])
    value = reader.get_tensor(name)
    #value = truncate_value(variable, value, reshape=reshape)
    yield variable, value

def load_snapshot(ckpt, session=None, var_list=None, reshape=False, scope=''):
  session = session or tf.get_default_session()
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vs = var_list or tf.trainable_variables()
  for variables in tqdm.tqdm(list(split_by_params(vs))[::-1]):
    values = [value for variable, value in grab_values(variables, reader, reshape=reshape, scope=scope)]
    assign_values(variables, values, session=session)

def restore(sess, ckpt, var_list=None, scope=''):
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  if var_list is None:
    var_list = tf.trainable_variables()
  for v in tqdm.tqdm(var_list):
    name = os.path.join(scope, v.name.split(':', 1)[0])
    if not reader.has_tensor(name):
      print('Warning: no such tensor', name)
      import pdb
      pdb.set_trace()
    else:
      value = reader.get_tensor(name)
      assign_values([v], [value], session=sess)


def run_next(sess, get_next, context, context_labels):
  print('Fetching...')
  labels, images = get_next(sess)
  d = {}
  print('Loading labels...')
  print(load2(context_labels, labels, session=sess))
  print('Loading images...')
  load2(context, images, session=sess)
  print('Loaded')
  if state.init:
    print('Initializing...')
    sess.run(tf.global_variables_initializer())
    print('Initialized.')
    state.init = None
  for i in range(params['train_iterations']):
    sess.run(state.train_op, d)
    result = sess.run(state.loss, d)
    print(result)
  return result

def load2(variable, value, session=None, timeout_in_ms=None):
  op = tf.assign(variable, value)
  session = session or tf.get_default_session()
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(op, options=options)

def load(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  ops = variable.initializer
  vals = dict([(variable.initializer.inputs[1], value)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(ops, vals, options=options)

class ResnetModelTest(tf.test.TestCase):

  def test_load_resnet18(self):
    network = resnet_model.resnet_v1(resnet_depth=50,
                                     num_classes=1001,
                                     data_format='channels_last')
    #input_bhw3 = tf.placeholder(tf.float32, [1, 224, 224, 3])
    context = tf.Variable(tf.zeros(shape=[params['batch_size'] // params['num_cores'], params['image_size'], params['image_size'], 3], name="context", dtype=tf.float32),
                          dtype=tf.float32, shape=[params['batch_size'] // params['num_cores'], params['image_size'], params['image_size'], 3], trainable=False)
    context_labels = tf.Variable([0] * params['batch_size'], name="context_labels", dtype=tf.int32, shape=[params['batch_size']], trainable=False)

    logits = network(inputs=context, is_training=True)

    sess = tf.Session(os.environ['TPU_NAME'] if 'TPU_NAME' in os.environ else None)
    lr = params['lr']
    state.optimizer = tf.train.AdamOptimizer(
      learning_rate=lr,
      beta1=params["beta1"],
      beta2=params["beta2"],
      epsilon=params["epsilon"])
    one_hot_labels = tf.one_hot(context_labels, params['num_label_classes'])
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels,
                                                    label_smoothing=params['label_smoothing'])
    state.loss = cross_entropy
    state.train_op = state.optimizer.minimize(state.loss, global_step=tf.train.get_global_step())
    state.init = True
    if False:
      ckpt = tf.train.latest_checkpoint('gs://gpt-2-poetry/checkpoint/resnet_imagenet_v1_fp32_20181001')
      #ckpt = tf.train.latest_checkpoint('./resnet_imagenet_v1_fp32_20181001')
      #restore(sess, ckpt, scope='resnet_model')
      load_snapshot(ckpt, sess, scope='resnet_model')
      #saver = tf.train.Saver()
      #print('Restoring...', ckpt)
      #saver.restore(sess, ckpt)
      #print('Restored.')
    #_ = sess.run(resnet_output, feed_dict={input_bhw3: np.random.randn(1, 224, 224, 3)})
    #print('TKTK', repr(_))
    get_next = iterate_imagenet(sess)
    state.start_time = time.time()
    state.prev_time = time.time()
    state.counter = 0
    while True:
      run_next(sess, get_next, context, context_labels)
      now = time.time()
      elapsed = now - state.prev_time
      total = now - state.start_time
      n = params['batch_size'] * params['train_iterations']
      state.counter += n
      print('[%.2fs | %d] %d examples in %.2fs (%.2f examples/sec)' % (total, state.counter, n, elapsed, n / elapsed))
      state.prev_time = now
    print('Done')
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
  tf.test.main()

def softmax(xs):
  return np.exp(xs) / sum(np.exp(xs))

