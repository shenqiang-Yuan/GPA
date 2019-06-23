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
            net = tf.layers.batch_normalization(net,training=not self.is_test,
                                            momentum=self.spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='comp_%d_spatbn_%d%s' % (self.comp_count, self.comp_idx, '_middle'))
            net = tf.nn.relu(net, name='comp_%d_activ_%d%s' % (self.comp_count, self.comp_idx, '_middle'))#所有未指定名字的relu operation 计算图上的RELU[0-XX]里面可以找到
            
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
        inputs,   # feature maps from preceding layer
        base_filters,    # num of filters internally in the component
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
            net = tf.layers.batch_normalization(net,training=not self.is_test,
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
            net = tf.layers.batch_normalization(net,training=not self.is_test,
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
            net = tf.layers.batch_normalization(net,training=not self.is_test,
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
                shortcut_blob = tf.layers.batch_normalization(shortcut_blob,training=not self.is_test,
                                            momentum=self.spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='shortcut_projection_%d_spatbn' % self.comp_count)
        net = tf.add_n([net, shortcut_blob],name='comp_%d_sum_%d' % (self.comp_count, self.comp_idx))
        
        self.comp_idx += 1
        net = tf.nn.relu(net, name='activate_com%d'%(self.comp_count))

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
            net = tf.layers.batch_normalization(net,training=not self.is_test,
                                            momentum=self.spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))
        net = tf.nn.relu(net, name='comp_%d_activ_%d'% (self.comp_count, self.comp_idx))

        net = self.add_conv(
            net,
            num_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            pads='same',
            is_decomposed=is_decomposed,
        )
        if spatial_batch_norm:
            net = tf.layers.batch_normalization(net,training=not self.is_test,
                                            momentum=self.spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx))

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
                shortcut_blob = tf.layers.batch_normalization(shortcut_blob,training=not self.is_test,
                                            momentum=self.spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='shortcut_projection_%d_spatbn' % self.comp_count)
        net = tf.add_n([net, shortcut_blob],name='comp_%d_sum_%d' % (self.comp_count, self.comp_idx))

        self.comp_idx += 1
        net = tf.nn.relu(net, name='activate_com%d'%(self.comp_count))

        # Keep track of number of high level components
        self.comp_count += 1
        return net

