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
import threading

from tensorflow.python import pywrap_tensorflow
import os
import time

from tensorflow.core.protobuf import config_pb2
import tqdm

from tensorflow.python.ops import gradients
import memory_saving_gradients

filename = 'gs://cloud-tpu-test-datasets/fake_imagenet/train-00000-of-01024'
filename = 'gs://gpt-2-poetry/data/imagenet/out/train-00000-of-01024'
filename = 'gs://gpt-2-poetry/data/imagenet/out/validation-00000-of-00128'

params = {}
#params['data_dir'] = 'gs://gpt-2-poetry/data/imagenet/out'
params['data_dir'] = 'gs://cloud-tpu-test-datasets/fake_imagenet'
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
params['shard'] = int(os.environ['SHARD']) if 'SHARD' in os.environ else -1
params['precision'] = os.environ['PRECISION'] if 'PRECISION' in os.environ else 'float32'
params['colocate_gradients_with_ops'] = bool(os.environ['COLOCATE_GRADIENTS']) if 'COLOCATE_GRADIENTS' in os.environ else True
params['ungate_gradients'] = bool(os.environ['UNGATE_GRADIENTS']) if 'UNGATE_GRADIENTS' in os.environ else False
params['num_train_images'] = 1281167
params['train_batch_size'] = params['batch_size']
params['eval_batch_size'] = params['batch_size']
import math
params['train_steps'] = math.ceil(32768*2983/params['train_batch_size'])
params['momentum'] = 0.9
params['weight_decay'] = 0.0001
params['enable_lars'] = False
params['base_learning_rate'] = 0.1

from pprint import pprint
pprint(params)

use_memory_saving_gradients = 'MEMORY_SAVING_GRADIENTS' in os.environ

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
        use_bfloat16 = params['precision'] == 'bfloat16'
        image = image_preprocessing_fn(image_bytes=image_bytes, is_training=True, image_size=224, use_bfloat16=use_bfloat16)
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

from official.resnet import imagenet_input

params['transpose_input'] = True

def iterate_imagenet():
  use_bfloat16 = params['precision'] == 'bfloat16'
  imagenet_train, imagenet_eval = [
    imagenet_input.ImageNetInput(is_training=True, data_dir=params['data_dir'], transpose_input=params['transpose_input'], cache=True,
                                 image_size=224, num_parallel_calls=4, include_background_label=True, use_bfloat16=use_bfloat16)
    for is_training in [True, False]]
  zz = imagenet_train.input_fn(params=params)
  it = zz.make_one_shot_iterator()
  nxt = it.get_next()
  def get_next(sess):
    images, labels = sess.run(nxt)
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
  print(load(context_labels, labels, session=sess))
  print('Loading images...')
  load(context, images, session=sess)
  print('Loaded')
  if state.init:
    print('Initializing...')
    sess.run(tf.global_variables_initializer())
    print('Initialized.')
    state.init = None
  result = sess.run(state.loss, d)
  print('start loss', result)
  start_time = time.time()
  n = params['batch_size']
  examples = 0
  for i in range(params['train_iterations']):
    sess.run(state.train_op, d)
    examples += n
  elapsed = time.time() - start_time
  print('%d examples in %.2fs (%.2f ex/sec)' % (examples, elapsed, examples / elapsed))
  result = sess.run(state.loss, d)
  print('end loss', result)
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

args = Namespace()
args.allow_growth = False
args.allow_soft_placement = True
args.disable_layout_optimizer = False
args.no_report_tensor_allocations_upon_oom = True

def main():
  timeout = 600000
  config = config_pb2.ConfigProto(operation_timeout_in_ms=timeout)
  config.allow_soft_placement = False
  if args.allow_growth:
    config.gpu_options.allow_growth = True
  if args.allow_soft_placement:
    config.allow_soft_placement = True
  if args.disable_layout_optimizer:
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
  options = config_pb2.RunOptions(report_tensor_allocations_upon_oom=(not args.no_report_tensor_allocations_upon_oom))
  target = os.environ['TPU_NAME'] if 'TPU_NAME' in os.environ else None
  sess = tf.Session(target=target, config=config)
  state.cores = sess.list_devices()[2:10]
  i = params['shard']
  if i >= 0:
    with tf.device(state.cores[i].name):
      shard(sess, i)
  else:
    shard(sess, 0)


# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def get_lr_schedule(train_steps, num_train_images, train_batch_size):
  """learning rate schedule."""
  steps_per_epoch = np.floor(num_train_images / train_batch_size)
  train_epochs = train_steps / steps_per_epoch
  return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(5 / 90 * train_epochs)),
      (0.1, np.floor(30 / 90 * train_epochs)),
      (0.01, np.floor(60 / 90 * train_epochs)),
      (0.001, np.floor(80 / 90 * train_epochs))
  ]


def learning_rate_schedule(params, current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    params: Python dict containing parameters for this run.
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = params['base_learning_rate'] * (
      params['train_batch_size'] / 256.0)
  lr_schedule = get_lr_schedule(
      train_steps=params['train_steps'],
      num_train_images=params['num_train_images'],
      train_batch_size=params['train_batch_size'])
  decay_rate = (scaled_lr * lr_schedule[0][0] *
                current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def shard(sess, i):
  scope = 'resnet_model'
  prefix = 'core%04d' % i
  with tf.variable_scope(prefix + '/' + scope, reuse=tf.AUTO_REUSE):
    shape = [params['batch_size'] // params['num_cores'], params['image_size'], params['image_size'], 3]
    if params['transpose_input']:
      shape = [params['batch_size'] * params['image_size'] * params['image_size'] * 3]
    if params['precision'] == 'bfloat16':
      with tf.tpu.bfloat16_scope():
        print('Using bfloat16')
        network = resnet_model.resnet_v1(resnet_depth=50,
                                         num_classes=1001,
                                         data_format='channels_last')
        context = tf.Variable(
          tf.zeros(shape=shape, name="context", dtype=tf.bfloat16),
          dtype=tf.bfloat16,
          shape=shape, trainable=False)
        context_labels = tf.Variable([0] * params['batch_size'], name="context_labels", dtype=tf.int32,
                                     shape=[params['batch_size']], trainable=False)

        features = tf.reshape(context, [params['image_size'], params['image_size'], 3, -1])
        features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

        # Normalize the image to zero mean and unit variance.
        features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
        features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

        logits = network(inputs=features, is_training=True)
        logits = tf.cast(logits, tf.float32)
    else:
      network = resnet_model.resnet_v1(resnet_depth=50,
                                       num_classes=1001,
                                       data_format='channels_last')
      context = tf.Variable(
        tf.zeros(shape=shape,
                 name="context", dtype=tf.float32),
        dtype=tf.float32,
        shape=shape,
        trainable=False)
      context_labels = tf.Variable([0] * params['batch_size'], name="context_labels", dtype=tf.int32,
                                   shape=[params['batch_size']], trainable=False)

      features = tf.reshape(context, [params['image_size'], params['image_size'], 3, -1])
      features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

      # Normalize the image to zero mean and unit variance.
      features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
      features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

      logits = network(inputs=features, is_training=True)

    global_step = tf.train.get_or_create_global_step()

    if False:
      lr = params['lr']
      #import pdb; pdb.set_trace()
      optimizer = tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=params["beta1"],
        beta2=params["beta2"],
        epsilon=params["epsilon"])
    else:
      # Compute the current epoch and associated learning rate from global_step.
      steps_per_epoch = params['num_train_images'] / params['train_batch_size']
      current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
      # LARS is a large batch optimizer. LARS enables higher accuracy at batch 16K
      # and larger batch sizes.
      if params['enable_lars']:
        learning_rate = 0.0
        optimizer = lars_util.init_lars_optimizer(current_epoch, params)
      else:
        learning_rate = learning_rate_schedule(params, current_epoch)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'],
            use_nesterov=True)

    one_hot_labels = tf.one_hot(context_labels, params['num_label_classes'])
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels,
                                                    label_smoothing=params['label_smoothing'])
      
  path = scope
  if prefix is not None:
    path = prefix + '/' + path
  global_vars = [v for v in tf.global_variables() if v.name.startswith(path + '/')]
  all_vars = [v for v in tf.trainable_variables() if v.name.startswith(path + '/')]
  def should_train_variable(v):
    return True
  train_vars = [v for v in all_vars if should_train_variable(v)]

  # Add weight decay to the loss for non-batch-normalization variables.
  if params['enable_lars']:
    state.loss = cross_entropy
  else:
    state.loss = cross_entropy + params['weight_decay'] * tf.add_n([
        tf.nn.l2_loss(v)
        for v in train_vars
        if 'batch_normalization' not in v.name
    ])

  colocate_gradients_with_ops = params['colocate_gradients_with_ops']
  gate_gradients=None
  if params['ungate_gradients']:
    gate_gradients=tf.train.Optimizer.GATE_NONE

  if False:
    if use_memory_saving_gradients:
      grads = memory_saving_gradients.gradients
      grads = grads(state.loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
    else:
      grads = gradients.gradients(state.loss, train_vars, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
    grads = list(zip(grads, train_vars))
    grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads]  # replace disconnected gradients with zeros
    #global_step = tf.train.get_global_step()
    #state.train_op = optimizer.minimize(state.loss, global_step=global_step)
    #state.train_op = optimizer.apply_gradients(grads, global_step=global_step)

  # Batch normalization requires UPDATE_OPS to be added as a dependency to
  # the train operation.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    state.train_op = optimizer.minimize(state.loss, global_step, colocate_gradients_with_ops=colocate_gradients_with_ops, gate_gradients=gate_gradients)
    #state.train_op = optimizer.apply_gradients(grads, global_step=global_step)

  state.init = True
  if False:
    ckpt = tf.train.latest_checkpoint('gs://gpt-2-poetry/checkpoint/resnet_imagenet_v1_fp32_20181001')
    # ckpt = tf.train.latest_checkpoint('./resnet_imagenet_v1_fp32_20181001')
    # restore(sess, ckpt, scope='resnet_model')
    load_snapshot(ckpt, sess, scope='resnet_model')
    # saver = tf.train.Saver()
    # print('Restoring...', ckpt)
    # saver.restore(sess, ckpt)
    # print('Restored.')
  # _ = sess.run(resnet_output, feed_dict={input_bhw3: np.random.randn(1, 224, 224, 3)})
  # print('TKTK', repr(_))
  get_next = iterate_imagenet()
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


class ResnetModelTest(tf.test.TestCase):

  def test_load_resnet18(self):
    main()

if __name__ == '__main__':
  tf.test.main()

def softmax(xs):
  return np.exp(xs) / sum(np.exp(xs))

