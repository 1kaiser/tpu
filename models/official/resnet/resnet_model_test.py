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

filename = 'gs://cloud-tpu-test-datasets/fake_imagenet/train-00000-of-01024'
filename = 'gs://gpt-2-poetry/data/imagenet/out/train-00000-of-01024'
filename = 'gs://gpt-2-poetry/data/imagenet/out/validation-00000-of-00128'

def iterate_imagenet(sess):
  image_preprocessing_fn = resnet_preprocessing.preprocess_image
  ds = tf.data.TFRecordDataset(filename)
  keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, ''), 'image/class/label': tf.FixedLenFeature([], tf.int64, -1), }
  def parse(value):
    return tf.parse_single_example(value, keys_to_features)
  result = ds.map(parse)
  tf_iterator = tf.data.Iterator.from_structure(result.output_types, result.output_shapes)
  init = tf_iterator.make_initializer(result)
  sess.run(init)
  def get_next(sess):
    parsed = sess.run(tf_iterator.get_next())
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image = image_preprocessing_fn(image_bytes=image_bytes, is_training=True, image_size=224, use_bfloat16=False)
    label = tf.cast(tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)
    image_result = sess.run(image)
    label_result = sess.run(label)
    return image_result, label_result
  return get_next

from tensorflow.python import pywrap_tensorflow
import os

from tensorflow.core.protobuf import config_pb2
import tqdm

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

params = {}
params['label_smoothing'] = 0.1
params['num_label_classes'] = 1001
params['beta1'] = 0.9
params['beta2'] = 0.999
params['epsilon'] = 1e-9
#params['lr'] = 0.245
params['lr'] = 0.000055

def run_next(sess, logits, get_next, input_bhw3):
  img, lbl = get_next(sess)
  print(lbl)
  d = {input_bhw3: img.reshape([-1, 224, 224, 3])}
  labels = [lbl]
  one_hot_labels = tf.one_hot(labels, params['num_label_classes'])
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels,
                                                  label_smoothing=params['label_smoothing'])
  loss = cross_entropy
  train_op = state.optimizer.minimize(loss, global_step=tf.train.get_global_step())
  #result = sess.run(loss, d)
  if state.init:
    #import pdb; pdb.set_trace()
    sess.run(tf.global_variables_initializer())
    state.init = None
  sess.run(train_op, d)
  result = sess.run(loss, d)
  return result

class ResnetModelTest(tf.test.TestCase):

  def test_load_resnet18(self):
    network = resnet_model.resnet_v1(resnet_depth=50,
                                     num_classes=1001,
                                     data_format='channels_last')
    input_bhw3 = tf.placeholder(tf.float32, [1, 224, 224, 3])
    resnet_output = network(inputs=input_bhw3, is_training=True)

    sess = tf.Session(os.environ['TPU_NAME'] if 'TPU_NAME' in os.environ else None)
    lr = params['lr']
    state.optimizer = tf.train.AdamOptimizer(
      learning_rate=lr,
      beta1=params["beta1"],
      beta2=params["beta2"],
      epsilon=params["epsilon"])
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
    #import pdb
    #pdb.set_trace()
    get_next = iterate_imagenet(sess)
    logits = resnet_output
    while True:
      print(run_next(sess, logits, get_next, input_bhw3))
      #import pdb
      #pdb.set_trace()
    #import pdb
    #pdb.set_trace()

if __name__ == '__main__':
  tf.test.main()

def softmax(xs):
  return np.exp(xs) / sum(np.exp(xs))

