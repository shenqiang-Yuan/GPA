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

from video_model import VideoModelBuilder
import tensorflow as tf
# For more depths, add the block config here
BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 3, 4, 3),
    34: (3, 4, 6, 3),
}


def create_model(
    data,
    model_name,
    model_depth,
    num_labels,
    num_channels,
    crop_size,
    clip_length,
    is_test=True,
    no_loss=True,
):
    if model_name == 'r2d' or model_name == 'r2df':
        if model_name == 'r2d':
            creator = create_r2d
            conv1_kernel_length = clip_length
            final_temporal_kernel = 1
        else:
            creator = create_r2df
            conv1_kernel_length = 1
            final_temporal_kernel = clip_length
        last_out = creator(
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_bias=1,
            no_loss=no_loss,
            final_spatial_kernel=7 if crop_size == 112 else 14,
            model_depth=model_depth,
            conv1_kernel_length=conv1_kernel_length,
            final_temporal_kernel=final_temporal_kernel,
        )
    elif model_name[0:2] == 'mc' or model_name[0:3] == 'rmc':
        # model_name = 'mc%d','rmc%d', %d: 2,3,4,5
        if model_name[0:2] == 'mc':
            mc_level = int(model_name[2])
            temporal_kernel = [8, 8, 4, 2]
            creator = create_mcx
        else:
            mc_level = int(model_name[3])
            temporal_kernel = [1, 1, 2, 4]
            creator = create_rmcx
        last_out = creator(
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_loss=no_loss,
            no_bias=1,
            final_spatial_kernel=7 if crop_size == 112 else 14,
            final_temporal_kernel=int(clip_length / 8) *
            temporal_kernel[mc_level - 2],
            model_depth=model_depth,
            mc_level=mc_level,
        )
    elif model_name == 'r3d' or model_name == 'r2plus1d':
        last_out = create_r3d(
            data=data,
            num_labels=num_labels,
            num_input_channels=num_channels,
            is_test=is_test,
            no_bias=1,
            no_loss=no_loss,
            final_spatial_kernel=7 if crop_size == 112 else 14,
            final_temporal_kernel=int(clip_length / 8),
            model_depth=model_depth,
            is_decomposed=(model_name == 'r2plus1d'),
        )
    else:
        raise NameError
    return last_out


# 3d or (2+1)d resnets, input 3 x t*8 x 112 x 112
# the final conv output is 512 * t * 7 * 7
def create_r3d(
    data,
    num_labels,
    num_input_channels=3,
    label=None,
    is_test=0,
    no_loss=0,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    is_decomposed=False,
    spatial_bn_mom=0.9,
):
    net = tf.identity(data, name='input_data')
    source = "source_node"

    # conv1 + maxpool
    if not is_decomposed:
        net = tf.layers.conv3d(net, 
                         filters=64, 
                         kernel_size=[3, 7, 7], 
                         strides=[1, 2, 2],
                         padding='same',
                         use_bias=not no_bias,
                         name='conv1')
        
    else:
        net = tf.layers.conv3d(net, 
                         filters=45,
                         kernel_size=[1, 7, 7], 
                         strides=[1, 2, 2],
                         padding='same',
                         use_bias=not no_bias,
                         name='conv1_middle')
        net = tf.layers.batch_normalization(net, training=not is_test,
                                            momentum=spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='conv1_middle_spatbn_relu')
        net = tf.nn.relu(net, name='conv1_middle_spatbn_activations')
        net = tf.layers.conv3d(net, 
                         filters=64,
                         kernel_size=[3, 1, 1], 
                         strides=[1, 1, 1],
                         padding='same',
                         use_bias=not no_bias,
                         name='conv1')
    net = tf.layers.batch_normalization(net,training=not is_test,
                                            momentum=spatial_bn_mom,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            name='conv1_spatbn_relu')
    net = tf.nn.relu(net, name='conv1_spatbn_activations')
    tf.add_to_collection(source, net)
    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(use_bias=not no_bias,
                                is_test=is_test,
                                spatial_bn_mom=spatial_bn_mom)
    
    # conv_2x
    for _ in range(n1):
        net = builder.add_simple_block(net,64, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    # conv_3x
    net = builder.add_simple_block(
        net, 128, down_sampling=True, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    for _ in range(n2 - 1):
        net = builder.add_simple_block(net, 128, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    # conv_4x
    net = builder.add_simple_block(
        net, 256, down_sampling=True, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    for _ in range(n3 - 1):
        net = builder.add_simple_block(net, 256, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    # conv_5x
    net = builder.add_simple_block(
        net, 512, down_sampling=True, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    for _ in range(n4 - 1):
       net = builder.add_simple_block(net, 512, is_decomposed=is_decomposed)

    # Final layers
    final_avg = tf.layers.average_pooling3d(net,
                                            pool_size=[
                                            final_temporal_kernel,
                                            final_spatial_kernel,
                                            final_spatial_kernel],
                                            strides=[1, 1, 1],
                                            name='final_avg')
    tf.add_to_collection(source, final_avg)
    net = tf.layers.flatten(final_avg, name='flatten')
    
    last_out = tf.layers.dense(net, num_labels,use_bias=True,name='last_out_L{}'.format(num_labels))
    tf.add_to_collection(source, last_out)

    if no_loss:
        return last_out

    # If we create model for training, use logits-with-softmax-loss
    if (label is not None):
        if len(label.shape)!=1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=label), axis=0)
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=last_out, labels=label), axis=0)
        return (last_out, loss)
    else:
        # For features extraction, we return logits
        return last_out

# 2d resnet18, input 3 x t*8 x 112 x 112
# perform 3D convolution but all temporal_kernel always 1
# just make average pool along whole temporal dimensions
def create_r2df(
    data,
    num_labels,
    num_input_channels=3,
    label=None,
    is_test=0,
    no_loss=0,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    conv1_kernel_length=1,
    spatial_bn_mom=0.9,
):
    assert conv1_kernel_length == 1
    net = tf.identity(data, name='input_data')
    source = "source_node"
    net = tf.layers.conv3d(net,
                           filters=64,
                           kernel_size=[conv1_kernel_length, 7, 7],
                           strides=[1, 2, 2],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv1')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=spatial_bn_mom,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv1_spatbn_relu')

    net = tf.nn.relu(net, name='conv1_activate_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(use_bias=not no_bias, is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        net = builder.add_simple_block(net, 64, is_real_3d=False)

    # conv_3x
    net = builder.add_simple_block(
        net, 128, down_sampling=True, is_real_3d=False)
    for _ in range(n2 - 1):
        net = builder.add_simple_block(net, 128, is_real_3d=False)

    # conv_4x
    net = builder.add_simple_block(
        net, 256, down_sampling=True, is_real_3d=False)
    for _ in range(n3 - 1):
        net = builder.add_simple_block(net, 256, is_real_3d=False)

    # conv_5x
    net = builder.add_simple_block(
        net, 512, down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        net = builder.add_simple_block(net, 512, is_real_3d=False)

    # Final layers
    final_avg = tf.layers.average_pooling3d(net,
                                            pool_size=[
                                                final_temporal_kernel,
                                                final_spatial_kernel,
                                                final_spatial_kernel],
                                            strides=[1, 1, 1],
                                            name='final_avg')
    # Final dimension of the "image" is reduced to 7x7
    net = tf.layers.flatten(final_avg, name='flatten')

    last_out = tf.layers.dense(net, num_labels, use_bias=True, name='last_out_L{}'.format(num_labels))

    if no_loss:
        # For inference, we just return softmax
        softmax = tf.nn.softmax(last_out, name='softmax')
        return softmax

    # If we create model for training, use logits-with-softmax-loss
    if (label is not None):
        if len(label.shape)!=1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=label), axis=0)
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=last_out, labels=label), axis=0)
        return (last_out, loss)
    else:
        # For features extraction, we return logits
        return last_out


# bottom 3D, Top 2D
def create_mcx(
    data,
    num_labels,
    num_input_channels=3,
    label=None,
    is_test=0,
    no_loss=0,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    mc_level=2,
    spatial_bn_mom=0.9,
):
    assert mc_level >= 2 and mc_level <= 5
    net = tf.identity(data, name='input_data')
    # conv1 + maxpool
    net = tf.layers.conv3d(net,
                           filters=64,
                           kernel_size=[3, 7, 7],
                           strides=[1, 2, 2],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv1')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=spatial_bn_mom,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv1_spatbn_relu')

    net = tf.nn.relu(net, name='conv1_activate_relu')


    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(use_bias=not no_bias, is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        net = builder.add_simple_block(
            net, 64, is_real_3d=True if mc_level > 2 else False)

    # conv_3x
    net = builder.add_simple_block(
        net, 128, down_sampling=True,
        is_real_3d=True if mc_level > 3 else False)
    for _ in range(n2 - 1):
        net = builder.add_simple_block(
            net, 128, is_real_3d=True if mc_level > 3 else False)

    # conv_4x
    net = builder.add_simple_block(
        net, 256, down_sampling=True,
        is_real_3d=True if mc_level > 4 else False)
    for _ in range(n3 - 1):
        net = builder.add_simple_block(
            net, 256,
            is_real_3d=True if mc_level > 4 else False)

    # conv_5x
    net = builder.add_simple_block(net, 512, down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        net = builder.add_simple_block(net, 512, is_real_3d=False)

    # Final layers
    final_avg = tf.layers.average_pooling3d(net,
                                            pool_size=[
                                            final_temporal_kernel,
                                            final_spatial_kernel,
                                            final_spatial_kernel],
                                            strides=[1, 1, 1],
                                            name='final_avg')
    # Final dimension of the "image" is reduced to 7x7
    net = tf.layers.flatten(final_avg, name='flatten')

    last_out = tf.layers.dense(net, num_labels, use_bias=True, name='last_out_L{}'.format(num_labels))

    if no_loss:
        # For inference, we just return softmax
        softmax = tf.nn.softmax(last_out,name='softmax')
        return softmax

    # If we create model for training, use logits-with-softmax-loss
    if (label is not None):
        if len(label.shape)!=1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=label), axis=0)
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=last_out, labels=label), axis=0)
        return (last_out, loss)
    else:
        # For features extraction, we return logits
        return last_out

# only first layer collapses the entire temporal information,
# which preclude any temporal reasoning to happen in subsequent layers
# at first layer: conv1_kernel_length=temporal_kernel, in_channel=3,
# here 3D conv equals to (W,H) of 2D conv with in_channel=3*clip_length.
# Later layers all temporal_kernel=1

# NOTE: the clip_length is fixed for the first conv layer if you want to resuse this pretrained parameter.

def create_r2d(
    data,
    num_labels,
    num_input_channels=3,
    label=None,
    is_test=0,
    no_loss=0,
    no_bias=0,
    final_spatial_kernel=7,
    model_depth=18,
    conv1_kernel_length=8,
    final_temporal_kernel=1,
    spatial_bn_mom=0.9,
):
    assert final_temporal_kernel == 1
    net = tf.identity(data, name='input_data')
    source = "source_node"

    # conv1 + maxpool
    # in caffe model here only padding for spatial, but no padding for temporal
    # if we simply use padding='same',the temporal will be auto padded.
    # so we pre-pad manually and then set padding='valid'
    paddings = tf.constant([[0, 0], [0, 0], [3, 3], [3, 3], [0, 0]])
    net = tf.pad(net, paddings, 'CONSTANT')
    net = tf.layers.conv3d(net,
                           filters=64,
                           kernel_size=[conv1_kernel_length, 7, 7],
                           strides=[1, 2, 2],
                           padding='valid',
                           use_bias=not no_bias,
                           name='conv1')

    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=spatial_bn_mom,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv1_spatbn_relu')

    net = tf.nn.relu(net, name='conv1_activate_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...

    builder = VideoModelBuilder(use_bias=not no_bias,
                                is_test=is_test,
                                spatial_bn_mom=spatial_bn_mom)
    # conv_2x
    for _ in range(n1):
        net = builder.add_simple_block(net, 64, is_real_3d=False)

    # conv_3x
    net = builder.add_simple_block(net, 128, down_sampling=True, is_real_3d=False)
    for _ in range(n2 - 1):
        net = builder.add_simple_block(net, 128, is_real_3d=False)

    # conv_4x
    net = builder.add_simple_block(net, 256, down_sampling=True, is_real_3d=False)
    for _ in range(n3 - 1):
        net = builder.add_simple_block(net, 256, is_real_3d=False)

    # conv_5x
    net = builder.add_simple_block(net, 512, down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        net = builder.add_simple_block(net, 512, is_real_3d=False)

    # Final layers
    final_avg = tf.layers.average_pooling3d(net,
                                            pool_size=[
                                            final_temporal_kernel,
                                            final_spatial_kernel,
                                            final_spatial_kernel],
                                            strides=[1, 1, 1],
                                            name='final_avg')
    # Final dimension of the "image" is reduced to 7x7
    net = tf.layers.flatten(final_avg, name='flatten')

    last_out = tf.layers.dense(net, num_labels, use_bias=True, name='last_out_L{}'.format(num_labels))

    if no_loss:
        # For inference, we just return softmax
        softmax = tf.nn.softmax(last_out,name='softmax')
        return softmax

    # If we create model for training, use logits-with-softmax-loss
    if (label is not None):
        if len(label.shape)!=1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=label), axis=0)
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=last_out, labels=label), axis=0)
        return (last_out, loss)
    else:
        # For features extraction, we return logits
        return last_out

# bottom 2D, Top 3D
def create_rmcx(
    data,
    num_labels,
    num_input_channels=3,
    label=None,
    is_test=0,
    no_loss=0,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    mc_level=2,
    spatial_bn_mom=0.9,
):
    assert mc_level >= 2 and mc_level <= 5
    # mc_level=2 means only first layer 2D is performed.
    # conv1 + maxpool
    net = tf.identity(data, name='input_data')
    net = tf.layers.conv3d(net,
                           filters=64,
                           kernel_size=[1, 7, 7],
                           strides=[1, 2, 2],
                           padding='same',
                           use_bias=not no_bias,
                           name='conv1')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=spatial_bn_mom,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv1_spatbn_relu')

    net = tf.nn.relu(net, name='conv1_activate_relu')



    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(use_bias=not no_bias, is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        net = builder.add_simple_block(
            net, 64, is_real_3d=True if mc_level <= 2 else False)

    # conv_3x
    net = builder.add_simple_block(
        net, 128, down_sampling=True,
        is_real_3d=True if mc_level <= 3 else False)
    for _ in range(n2 - 1):
        net = builder.add_simple_block(
            net, 128, is_real_3d=True if mc_level <= 3 else False)

    # conv_4x
    net = builder.add_simple_block(
        net, 256, down_sampling=True,
        is_real_3d=True if mc_level <= 4 else False)
    for _ in range(n3 - 1):
        net = builder.add_simple_block(
            net, 256, is_real_3d=True if mc_level <= 4 else False)

    # conv_5x
    net = builder.add_simple_block(
        net, 512, down_sampling=True, is_real_3d=True)
    for _ in range(n4 - 1):
        net = builder.add_simple_block(net, 512, is_real_3d=True)

    # Final layers
    final_avg = tf.layers.average_pooling3d(net,
                                            pool_size=[
                                            final_temporal_kernel,
                                            final_spatial_kernel,
                                            final_spatial_kernel],
                                            strides=[1, 1, 1],
                                            name='final_avg')
    # Final dimension of the "image" is reduced to 7x7
    net = tf.layers.flatten(final_avg, name='flatten')

    last_out = tf.layers.dense(net, num_labels, use_bias=True, name='last_out_L{}'.format(num_labels))

    if no_loss:
        # For inference, we just return softmax
        softmax = tf.nn.softmax(last_out,name='softmax')
        return softmax

    # If we create model for training, use logits-with-softmax-loss
    if (label is not None):
        if len(label.shape)!=1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_out, labels=label), axis=0)
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=last_out, labels=label), axis=0)
        return (last_out, loss)
    else:
        # For features extraction, we return logits
        return last_out


if __name__ == '__main__':
    # transform caffe model to tf model
    import pickle
    crop_size = 112
    clip_length = 16
    assert crop_size==112 or crop_size==224
    assert clip_length%8 == 0
    classes = 400
    """
    # rmcx_transformed
    with open('/home/pr606/YUAN/history/caffe2-R2plus1D/DownLoad_models/rmc2_d18_l16.pkl', 'rb') as fopen:
        blobs = pickle.load(fopen, encoding='latin1')
    if 'blobs' in blobs:
        blobs = blobs['blobs']
    i = 0
    for key, value in blobs.items():
        print(key, value.shape, i+1)
        i+=1
    data = tf.placeholder(shape=(None, clip_length, crop_size, crop_size, 3), dtype=tf.float32)
    labels = tf.placeholder(shape=(None,classes),dtype=tf.int32)
    
    mc_level = 2
    rmcx_list = [1, 1, 2, 4]
    Kt = rmcx_list[mc_level-2]*int(clip_length/8)
    result = create_rmcx(data,classes, label=labels,
                         is_test=0,final_temporal_kernel=Kt,
                         no_bias=1,
                         final_spatial_kernel=7 if crop_size == 112 else 14)
    
    variable_map={}
    i = 0
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        '''
        if 'L101' in var.name:
            continue
        '''
        variable_map[var.name.replace(":0", '')] = var
        print(var.name, var.shape, i+1)
        i+=1
    global_init = tf.global_variables_initializer()
    rgb_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    # tensorflow: w=(Kt,Kh,Kw,inchannel,outchannel)
    # caffe: w=(outchannel,inchannel,Kt,Kh,Kw)   transpose(2,3,4,1,0)

    with tf.Session() as sess:
        sess.run(global_init)
        for key, values in blobs.items():
            suffix = key[key.rfind('_') + 1:]
            name = key[0:key.rfind('_')]
            if suffix == 'riv':
                sess.run(tf.assign(variable_map[name + "/moving_variance"], values))
                continue
            if suffix == 'rm':
                sess.run(tf.assign(variable_map[name + "/moving_mean"], values))
                continue
            if suffix == 's':
                sess.run(tf.assign(variable_map[name + "/gamma"], values))
                continue
            if suffix == 'b':
                if key.endswith("L400_b"):
                    sess.run(tf.assign(variable_map[name + "/bias"], values))
                    continue
                else:
                    sess.run(tf.assign(variable_map[name + "/beta"], values))
                    continue
            if suffix == 'w':
                if key.endswith("L400_w"):
                    sess.run(tf.assign(variable_map[name + "/kernel"], values.transpose(1, 0)))
                    continue
                else:
                    sess.run(tf.assign(variable_map[name + "/kernel"], values.transpose(2, 3, 4, 1, 0)))
                    continue
        rgb_saver.save(sess, '/home/pr606/YUAN/history/tf-R2plus1D/pretrained-models/rmcx-b/rmc2.ckpt')
        pass
        """
    """
    # r3d_transformed
    with open('/home/pr606/YUAN/history/caffe2-R2plus1D/DownLoad_models/r3d_d18_l16.pkl', 'rb') as fopen:
        blobs = pickle.load(fopen, encoding='latin1')
    if 'blobs' in blobs:
        blobs = blobs['blobs']
    i = 0
    for key, value in blobs.items():
        print(key, value.shape, i + 1)
        i += 1
    data = tf.placeholder(shape=(None, clip_length, crop_size, crop_size, 3), dtype=tf.float32)
    labels = tf.placeholder(shape=(None, classes), dtype=tf.int32)

    result = create_r3d(data, classes, label=labels,
                        is_test=0,
                        no_bias=1,
                        no_loss=1,
                        final_spatial_kernel=7 if crop_size == 112 else 14,
                        final_temporal_kernel=int(clip_length / 8),
                        is_decomposed=False,
                        model_depth=18)
    variable_map = {}
    i = 0
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        '''
        if 'L101' in var.name:
            continue
        '''
        variable_map[var.name.replace(":0", '')] = var
        print(var.name, var.shape, i + 1)
        i += 1
    global_init = tf.global_variables_initializer()
    rgb_saver = tf.train.Saver(var_list=variable_map, reshape=True)
    with tf.Session() as sess:
        sess.run(global_init)
        for key, values in blobs.items():
            suffix = key[key.rfind('_') + 1:]
            name = key[0:key.rfind('_')]
            if suffix == 'riv':
                sess.run(tf.assign(variable_map[name + "/moving_variance"], values))
                continue
            if suffix == 'rm':
                sess.run(tf.assign(variable_map[name + "/moving_mean"], values))
                continue
            if suffix == 's':
                sess.run(tf.assign(variable_map[name + "/gamma"], values))
                continue
            if suffix == 'b':
                if key.endswith("L400_b"):
                    sess.run(tf.assign(variable_map[name + "/bias"], values))
                    continue
                else:
                    sess.run(tf.assign(variable_map[name + "/beta"], values))
                    continue
            if suffix == 'w':
                if key.endswith("L400_w"):
                    sess.run(tf.assign(variable_map[name + "/kernel"], values.transpose(1, 0)))
                    continue
                else:
                    sess.run(tf.assign(variable_map[name + "/kernel"], values.transpose(2, 3, 4, 1, 0)))
                    continue
        rgb_saver.save(sess, '/home/pr606/YUAN/history/tf-R2plus1D/pretrained-models/r3d-18/resnet-18-l16.ckpt')
        pass
    """
    with open('/home/pr606/YUAN/history/caffe2-R2plus1D/DownLoad_models/r2.5d_d18_l16.pkl', 'rb') as fopen:
        blobs = pickle.load(fopen, encoding='latin1')
    if 'blobs' in blobs:
        blobs = blobs['blobs']
    i = 0
    for key, value in blobs.items():
        print(key, value.shape, i + 1)
        i += 1
    data = tf.placeholder(shape=(None, clip_length, crop_size, crop_size, 3), dtype=tf.float32)
    labels = tf.placeholder(shape=(None, classes), dtype=tf.int32)

    result = create_r3d(data, classes, label=labels,
                        is_test=0,
                        no_bias=1,
                        no_loss=1,
                        final_spatial_kernel=7 if crop_size == 112 else 14,
                        final_temporal_kernel=int(clip_length / 8),
                        is_decomposed=True,
                        model_depth=18)
    variable_map = {}
    i = 0
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        '''
        if 'L101' in var.name:
            continue
        '''
        variable_map[var.name.replace(":0", '')] = var
        print(var.name, var.shape, i + 1)
        i += 1
    global_init = tf.global_variables_initializer()
    rgb_saver = tf.train.Saver(var_list=variable_map, reshape=True)
    with tf.Session() as sess:
        sess.run(global_init)
        for key, values in blobs.items():
            suffix = key[key.rfind('_') + 1:]
            name = key[0:key.rfind('_')]
            if suffix == 'riv':
                sess.run(tf.assign(variable_map[name + "/moving_variance"], values))
                continue
            if suffix == 'rm':
                sess.run(tf.assign(variable_map[name + "/moving_mean"], values))
                continue
            if suffix == 's':
                sess.run(tf.assign(variable_map[name + "/gamma"], values))
                continue
            if suffix == 'b':
                if key.endswith("L400_b"):
                    sess.run(tf.assign(variable_map[name + "/bias"], values))
                    continue
                else:
                    sess.run(tf.assign(variable_map[name + "/beta"], values))
                    continue
            if suffix == 'w':
                if key.endswith("L400_w"):
                    sess.run(tf.assign(variable_map[name + "/kernel"], values.transpose(1, 0)))
                    continue
                else:
                    sess.run(tf.assign(variable_map[name + "/kernel"], values.transpose(2, 3, 4, 1, 0)))
                    continue
        rgb_saver.save(sess, '/home/pr606/YUAN/history/tf-R2plus1D/pretrained-models/r2plus1-18/r2plus1.ckpt')
        pass
    
