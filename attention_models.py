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

# adopted from @package resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
from tensorflow.python.ops import init_ops

logging.basicConfig()
log = logging.getLogger("video_model")
log.setLevel(logging.DEBUG)


class VideoModelBuilder(object):
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(self, use_bias, is_test, spatial_bn_mom=0.9):
        self.comp_count = 0
        self.comp_idx = 0
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.use_bias = use_bias
        self.type = 'att'
        self.fusion = 'mul'
        self.Gc = {'ratio':1.0/8,
                   'type':['avg','att'],
                   'fusion':['add', 'mul'],
                   'position':['com_count1']}

    def add_conv(
            self,
            inputs,
            out_filters,
            kernels,
            strides=[1, 1, 1],
            pads='same',
            is_decomposed=False,  # set this to be True for (2+1)D conv
    ):
        self.comp_idx += 1
        in_filters = inputs.get_shape().as_list()[-1]

        if is_decomposed:
            i = 3 * in_filters * out_filters * kernels[1] * kernels[2]
            i /= in_filters * kernels[1] * kernels[2] + 3 * out_filters
            middle_filters = int(i)

            log.info("Number of middle filters: {}".format(middle_filters))
            net = tf.layers.conv3d(inputs,
                                   filters=middle_filters,
                                   kernel_size=[1, kernels[1], kernels[2]],
                                   strides=[1, strides[1], strides[2]],
                                   padding=pads,
                                   use_bias=self.use_bias,
                                   name='comp_%d_conv_%d_middle' % (self.comp_count, self.comp_idx))
            net = tf.layers.batch_normalization(net, training=not self.is_test,
                                                momentum=self.spatial_bn_mom,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                name='comp_%d_spatbn_%d%s' % (
                                                self.comp_count, self.comp_idx, '_middle'))
            net = tf.nn.relu(net, name='comp_%d_activ_%d%s' % (
            self.comp_count, self.comp_idx, '_middle'))  # 所有未指定名字的relu operation 计算图上的RELU[0-XX]里面可以找到

            net = tf.layers.conv3d(net,
                                   filters=out_filters,
                                   kernel_size=[kernels[0], 1, 1],
                                   strides=[strides[0], 1, 1],
                                   padding=pads,
                                   use_bias=self.use_bias,
                                   name='comp_%d_conv_%d' % (self.comp_count, self.comp_idx))
        else:
            net = tf.layers.conv3d(inputs,
                                   filters=out_filters,
                                   kernel_size=kernels,
                                   strides=strides,
                                   padding=pads,
                                   use_bias=self.use_bias,
                                   name='comp_%d_conv_%d' % (self.comp_count, self.comp_idx))
        return net

    '''
    Add a "bottleneck" component which can be 2d, 3d, (2+1)d
    '''

    def add_bottleneck(
            self,
            inputs,  # feature maps from preceding layer
            base_filters,  # num of filters internally in the component
            output_filters,  # num of feature maps to output
            down_sampling=False,
            spatial_batch_norm=True,
            is_decomposed=False,
            is_real_3d=True,
    ):
        if is_decomposed:
            # decomposition can only be applied to 3d conv
            assert is_real_3d

        self.comp_idx = 0
        shortcut_blob = inputs
        input_filters = inputs.shape.as_list()[-1]
        # 1x1x1
        net = self.add_conv(
            inputs,
            base_filters,
            kernels=[1, 1, 1],
            pads='same',
            is_decomposed=is_decomposed,
        )

        if spatial_batch_norm:
            net = tf.layers.batch_normalization(net, training=not self.is_test,
                                                momentum=self.spatial_bn_mom,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))

        net = tf.nn.relu(net, name='comp_%d_base_activ_%d' % (self.comp_count, self.comp_idx))

        if down_sampling:
            if is_real_3d:
                use_striding = [2, 2, 2]
            else:
                use_striding = [1, 2, 2]
        else:
            use_striding = [1, 1, 1]

        # 3x3x3 (note the pad, required for keeping dimensions)
        net = self.add_conv(
            net,
            base_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            strides=use_striding,
            pads='same',
            is_decomposed=is_decomposed,
        )

        if spatial_batch_norm:
            net = tf.layers.batch_normalization(net, training=not self.is_test,
                                                momentum=self.spatial_bn_mom,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))
        net = tf.nn.relu(net, name='comp_%d_base_activ_%d' % (self.comp_count, self.comp_idx))

        # 1x1x1
        net = self.add_conv(
            net,
            output_filters,
            kernels=[1, 1, 1],
            pads='same',
            is_decomposed=is_decomposed,
        )
        if spatial_batch_norm:
            net = tf.layers.batch_normalization(net, training=not self.is_test,
                                                momentum=self.spatial_bn_mom,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))

        # Summation with input signal (shortcut)
        # If we need to increase dimensions (feature maps), need to
        # do do a projection for the short cut
        if (output_filters != input_filters):
            shortcut_blob = tf.layers.conv3d(shortcut_blob,
                                             filters=output_filters,
                                             kernel_size=[1, 1, 1],
                                             strides=use_striding,
                                             padding='same',
                                             use_bias=bool(self.no_bias),
                                             name='shortcut_projection_%d' % self.comp_count)

            if spatial_batch_norm:
                shortcut_blob = tf.layers.batch_normalization(shortcut_blob, training=not self.is_test,
                                                              momentum=self.spatial_bn_mom,
                                                              epsilon=0.001,
                                                              center=True,
                                                              scale=True,
                                                              name='shortcut_projection_%d_spatbn' % self.comp_count)
        net = tf.add_n([net, shortcut_blob], name='comp_%d_sum_%d' % (self.comp_count, self.comp_idx))

        self.comp_idx += 1
        net = tf.nn.relu(net, name='activate_com%d' % (self.comp_count))

        # Keep track of number of high level components
        self.comp_count += 1
        return net

    '''
    Add a "simple_block" component which can be 2d, 3d, (2+1)d
    '''

    def add_simple_block(
            self,
            inputs,
            num_filters,
            down_sampling=False,
            spatial_batch_norm=True,
            is_decomposed=False,
            is_real_3d=True,
            only_spatial_downsampling=False,
    ):
        if is_decomposed:
            # decomposition can only be applied to 3d conv
            assert is_real_3d

        self.comp_idx = 0
        shortcut_blob = inputs
        input_filters = inputs.shape.as_list()[-1]
        if down_sampling:
            if is_real_3d:
                if only_spatial_downsampling:
                    use_striding = [1, 2, 2]
                else:
                    use_striding = [2, 2, 2]
            else:
                use_striding = [1, 2, 2]
        else:
            use_striding = [1, 1, 1]

        # 3x3x3
        net = self.add_conv(
            inputs,
            num_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            strides=use_striding,
            pads='same',
            is_decomposed=is_decomposed,
        )

        if spatial_batch_norm:
            net = tf.layers.batch_normalization(net, training=not self.is_test,
                                                momentum=self.spatial_bn_mom,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))
        net = tf.nn.relu(net, name='comp_%d_activ_%d' % (self.comp_count, self.comp_idx))

        net = self.add_conv(
            net,
            num_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            pads='same',
            is_decomposed=is_decomposed,
        )
        if spatial_batch_norm:
            net = tf.layers.batch_normalization(net, training=not self.is_test,
                                                momentum=self.spatial_bn_mom,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,

                                             name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))
        ''' 
        if 1<0:
            midc = self.Gc['ratio'] * num_filters
            net = self.Gc_uint(net, midc, name='attention_%d' % self.comp_count)
        '''
        if self.comp_count+1 == 4:
            net = self.GPA(net,64,index=1,v_nodes=192,reduce_factor=13,name='GPA1')
        if self.comp_count+1 == 7:
            net = self.GPA(net,64,index=1,v_nodes=64,reduce_factor=7,name='GPA2')
        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters) or down_sampling:
            shortcut_blob = tf.layers.conv3d(shortcut_blob,
                                             filters=num_filters,
                                             kernel_size=[1, 1, 1],
                                             strides=use_striding,
                                             padding='same',
                                             use_bias=self.use_bias,
                                             name='shortcut_projection_%d' % self.comp_count)

            if spatial_batch_norm:
                shortcut_blob = tf.layers.batch_normalization(shortcut_blob, training=not self.is_test,
                                                              momentum=self.spatial_bn_mom,
                                                              epsilon=0.001,
                                                              center=True,
                                                              scale=True,
                                                              name='shortcut_projection_%d_spatbn' % self.comp_count)
        net = tf.add_n([net, shortcut_blob], name='comp_%d_sum_%d' % (self.comp_count, self.comp_idx))

        self.comp_idx += 1
        net = tf.nn.relu(net, name='activate_com%d' % (self.comp_count))

        # Keep track of number of high level components
        self.comp_count += 1
        return net

    def Gc_uint(self, preblob, mid_channels, name=None):
        if name:
            pass
        else:
            name = preblob.name.replace(":0", '') + '_Gc'
        with tf.variable_scope(name):
            shape = preblob.shape.as_list()
            assert len(shape) == 5, 'only 3D attention supported'
            if self.type == 'avg':
                blob = tf.nn.max_pool3d(preblob,
                                        (1, shape[1], shape[2], shape[3], 1),
                                        strides=(1, 1, 1, 1, 1),
                                        padding='VALID',
                                        data_format='NDHWC',
                                        name='Context_modeling'
                                        )
                blob = tf.squeeze(blob, axis=(1, 2, 3), name=self.type)
            elif self.type == 'att':
                blob = tf.reshape(preblob,(-1,shape[4],shape[1]*shape[2]*shape[3]),name='flatten0')
                _blob = tf.layers.conv3d(preblob,1,(1,1,1),name='Context_modeling')
                _blob = tf.reshape(_blob,(-1,shape[1]*shape[2]*shape[3],1),name='flatten1')
                _blob = tf.nn.softmax(_blob,axis=1,name='scale')
                blob = tf.matmul(blob,_blob,name='dot_product')
                blob = tf.squeeze(blob,axis=-1,name=self.type)
            else:
                assert False, 'Unsupported Context Modeling type'

            w1 = tf.get_variable(name='se_fc1/kernel', shape=(shape[4], mid_channels))
            blob = tf.matmul(blob, w1, name='fc1')
            blob = tf.nn.relu(blob,name='fc1_relu')

            blob = tf.layers.batch_normalization(blob,momentum=0.88,name='LayerNorm')

            w2 = tf.get_variable(name='se_fc2/kernel', shape=(mid_channels, shape[4]))
            blob = tf.matmul(blob, w2, name='fc2')
            blob = tf.reshape(blob, (-1, 1, 1, 1, shape[4]), name='shape_matching')
            if self.fusion == 'mul':
                preblob = tf.multiply(preblob, blob, name='apply_attention%d' % self.comp_count)
            elif self.fusion == 'Add':
                preblob = tf.add(preblob,blob,name='apply_attention%d' % self.comp_count)
            else:
                assert self.fusion in self.Gc['fusion'],'Unsupported fusion type'
        return preblob

    def GPA(self, preblob, mid_channels, index, v_nodes=192, reduce_factor=13, name=None):
        if name:
            pass
        else:
            name = preblob.name.split('/')[-1].replace(":0", '') + '_GPA_%d'%index

        with tf.variable_scope(name):
            shape = preblob.shape.as_list()
            assert len(shape) == 5, 'only 3D attention supported'
            blob = tf.layers.conv3d(preblob,
                                    mid_channels,
                                    kernel_size=(1, 1, 1),
                                    strides=(1, 1, 1),
                                    padding='VALID',
                                    data_format='channels_last',
                                    use_bias=False,
                                    name='Dim_reducing'
                                    )

            _blob = tf.reshape(blob, (-1, shape[1] * shape[2] * shape[3], mid_channels), name='Coordinate_ignored')
            project_matrix = tf.layers.conv3d(preblob,
                                              v_nodes,
                                              kernel_size=(1, 1, 1),
                                              strides=(1, 1, 1),
                                              padding='VALID',
                                              data_format='channels_last',
                                              use_bias=False,
                                              name='matrixB_generated'
                                              )
            project_matrix = tf.reshape(project_matrix, (-1, shape[1] * shape[2] * shape[3], v_nodes), name='B')
            project_matrix = tf.nn.softmax(project_matrix, axis=1, name='Unitization')
            V = tf.matmul(tf.transpose(project_matrix, perm=(0, 2, 1)), _blob, name='projection') # NxC
            with tf.name_scope('global_reasoning'):
                use_kernel = (2 * reduce_factor * v_nodes) // (reduce_factor + v_nodes)
                use_stride = use_kernel // 2
                propagate_node = tf.layers.conv1d(V, mid_channels, (use_kernel,), strides=(use_stride,),
                                                  data_format='channels_last',
                                                  use_bias=False,
                                                  name='gathering_1th')
                propagate_node = tf.layers.batch_normalization(propagate_node, name='BN1')
                propagate_node = tf.nn.sigmoid(propagate_node, name='activate_gathering_1th')
                v_nodes = propagate_node.shape[1].value
                reduce_factor = reduce_factor//2
                use_kernel = (2 * reduce_factor * v_nodes) // (reduce_factor + v_nodes)
                use_stride = use_kernel // 2
                propagate_node = tf.layers.conv1d(propagate_node, shape[4], (use_kernel,), strides=(use_stride,),
                                                     data_format='channels_last',
                                                     use_bias=False,
                                                     name='gathering_2th')
                propagate_node = tf.layers.batch_normalization(propagate_node, name='BN2')
                propagate_node = tf.nn.sigmoid(propagate_node, name='activate_gathering_2th')
                v_nodes = propagate_node.shape[1].value
                attentions = tf.layers.max_pooling1d(propagate_node, pool_size=(v_nodes,), strides=(1,),name='max_pool')
            attentions = tf.reshape(attentions, shape=(-1, 1, 1, 1, shape[4]), name='shape_matching')
            # Y = tf.multiply(blob, attentions, name='apply_attention%d' % index)
            blob = tf.add(preblob, attentions, name='apply_attention%d' % index)

        return blob
