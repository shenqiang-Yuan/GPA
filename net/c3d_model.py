# Copyright 2018-present, Facebook, Inc.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


# implement C3D model with additional batch norm layers
# input 3 x 16 x 112 x 112
# reference model is here https://fburl.com/cfzvuwbj
def create_model(
    data,
    num_labels,
    num_input_channels=3,
    label=None,
    is_test=True,
    no_loss=True,
    no_bias=0,
    fc6_dim=4096,
    fc7_dim=4096,
    momentum=0.9,
    droupt=0.3
):
    # first conv layers
    net = tf.identity(data, name='input_data')
    net = tf.layers.conv3d(net,
                           filters=64,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv1a')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv1a_bn')
    net = tf.nn.relu(net, name='conv1a_spatbn_activations')
    net = tf.layers.max_pooling3d(net,(1,2,2),(1,2,2),name='pool1')
    # second conv layers
    net = tf.layers.conv3d(net,
                           filters=128,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv2a')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv2a_bn')
    net = tf.nn.relu(net, name='conv2a_spatbn_activations')
    net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2), name='pool2')

    # third conv layers
    net = tf.layers.conv3d(net,
                           filters=256,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv3a')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv3a_bn')
    net = tf.nn.relu(net, name='conv3a_spatbn_activations')
    net = tf.layers.conv3d(net,
                           filters=256,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv3b')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv3b_bn')
    net = tf.nn.relu(net, name='conv3b_spatbn_activations')
    net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2), name='pool3')

    # fourth conv layers
    net = tf.layers.conv3d(net,
                           filters=512,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv4a')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv4a_bn')
    net = tf.nn.relu(net, name='conv4a_spatbn_activations')
    net = tf.layers.conv3d(net,
                           filters=512,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv4b')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv4b_bn')
    net = tf.nn.relu(net, name='conv4b_spatbn_activations')
    net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2), name='pool4')

    # fifth conv layers
    net = tf.layers.conv3d(net,
                           filters=512,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv5a')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv5a_bn')
    net = tf.nn.relu(net, name='conv5a_spatbn_activations')
    net = tf.layers.conv3d(net,
                           filters=512,
                           kernel_size=[3, 3, 3],
                           strides=[1, 1, 1],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv5b')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=momentum,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv5b_bn')
    net = tf.nn.relu(net, name='conv5b_spatbn_activations')
    net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2), name='pool5')
    # Batch_size*1*3*3*512

    net = tf.layers.flatten(net,name='flatten')
    net = tf.layers.dense(net,fc6_dim,use_bias=True,name='fc6')
    net = tf.nn.relu(net, name='fc6_activations')
    if is_test:
        pass
    else:
        net = tf.layers.dropout(net,rate=droupt)
    net = tf.layers.dense(net, fc7_dim, use_bias=True, name='fc7')
    net = tf.nn.relu(net, name='fc7_activations')
    if is_test:
        pass
    else:
        net = tf.layers.dropout(net,rate=droupt)
    net = tf.layers.dense(net, num_labels, use_bias=True, name='last_out_L{}'.format(num_labels))

    if no_loss:
        return net

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        if len(label.shape)!=1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=label), axis=0)
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=label), axis=0)
        return (net, loss)
    else:
        # For inference, we just return softmax
        return net
