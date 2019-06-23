import tensorflow as tf
import numpy as np
import math


def Gaussian(inputs, in_channels=3, dimensions=3,theta=0.8, multi_scale=False,name='gaussian_filter'):
    '''
    :param inputs:
    :param name:
    :return:
    '''
    if multi_scale:
        Gaussian_kernel1 = Gaussian_template(theta=theta, accept_field=1, dimensions=dimensions)
        Gaussian_kernel2 = Gaussian_template(theta=0.5, accept_field=1, dimensions=dimensions)
        Gaussian_kernel3 = Gaussian_template(theta=0.3, accept_field=1, dimensions=dimensions)
        size=Gaussian_kernel1.shape
        dims=len(size)
        size = (1,)+size if dims==2 else size
        if in_channels==1:
            pass
        else:
            Gaussian_kernel1 = np.expand_dims(Gaussian_kernel1,axis=dims)
            Gaussian_kernel2 = np.expand_dims(Gaussian_kernel2,axis=dims)
            Gaussian_kernel3 = np.expand_dims(Gaussian_kernel3,axis=dims)
            _kernel1 = np.concatenate([Gaussian_kernel1 for _ in range(in_channels)], axis=dims)
            _kernel2 = np.concatenate([Gaussian_kernel2 for _ in range(in_channels)], axis=dims)
            _kernel3 = np.concatenate([Gaussian_kernel3 for _ in range(in_channels)], axis=dims)
    
        Gaussian1 = tf.get_variable(name=name+'_1', shape=size + (in_channels, 1), dtype=tf.float32,
                                initializer=tf.constant_initializer(_kernel1),trainable=False)
        Gaussian2 = tf.get_variable(name=name+'_2', shape=size + (in_channels, 1), dtype=tf.float32,
                                initializer=tf.constant_initializer(_kernel2),trainable=False)
        Gaussian3 = tf.get_variable(name=name+'_3', shape=size + (in_channels, 1), dtype=tf.float32,
                                initializer=tf.constant_initializer(_kernel3),trainable=False)
        filter_gaussian1 = tf.nn.conv3d(inputs, Gaussian1, (1, 1, 1, 1, 1), padding='SAME')
        filter_gaussian2 = tf.nn.conv3d(inputs, Gaussian2, (1, 1, 1, 1, 1), padding='SAME')
        filter_gaussian3 = tf.nn.conv3d(inputs, Gaussian3, (1, 1, 1, 1, 1), padding='SAME')
        filter_gaussian = tf.concat([filter_gaussian1, filter_gaussian2, filter_gaussian3], axis=-1)
    else:
        Gaussian_kernel = Gaussian_template(theta=theta, accept_field=1, dimensions=dimensions)
        size=Gaussian_kernel.shape
        dims=len(size)
        size = (1,)+size if dims==2 else size
        Gaussian = tf.get_variable(name=name, shape=size + (in_channels, 1), dtype=tf.float32,
                                initializer=tf.constant_initializer(Gaussian_kernel),trainable=False)
        filter_gaussian = tf.nn.conv3d(inputs, Gaussian, (1, 1, 1, 1, 1), padding='SAME')
    return filter_gaussian


def Batch_Norm(inputs,name='NORM'):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(inputs, axes=(0, 1, 2, 3), keep_dims=True, name="normalize_moments")
        Gamma = tf.constant(1.0, name="scale_factor", shape=mean.shape, dtype=tf.float32)
        Beta = tf.constant(0.0, name="offset_factor", shape=mean.shape, dtype=tf.float32)
        _inputs = tf.nn.batch_normalization(inputs, mean, variance, offset=Beta, scale=Gamma, variance_epsilon=1e-3)
    return _inputs


def get_temporal_kernel(size, in_channel, out_channel):
    if size == 3:
        laplace3 = np.array([[[[[1]]]], [[[[-2]]]], [[[[1]]]]], dtype=np.float32)
        _kernel3 = np.concatenate([laplace3 for _ in range(out_channel)], axis=4)
        if in_channel==1:
            kernel3=_kernel3
        else:
            kernel3 = np.concatenate([_kernel3 for _ in range(in_channel)], axis=3)
        return kernel3
    elif size == 5:
        laplace5 = np.array(
            [[[[[0.25]]]], [[[[0]]]], [[[[-0.5]]]], [[[[0]]]],
             [[[[0.25]]]]], dtype=np.float32)
        _kernel5 = np.concatenate([laplace5 for _ in range(out_channel)], axis=4)
        if in_channel==1:
            kernel5=_kernel5
        else:
            kernel5 = np.concatenate([_kernel5 for _ in range(in_channel)], axis=3)
        return kernel5
    elif size == 7:
        laplace7 = np.array(
            [[[[[1 / 9]]]], [[[[0]]]], [[[[0]]]], [[[[-2 / 9]]]],
             [[[[0]]]], [[[[0]]]], [[[[1 / 9]]]]], dtype=np.float32)
        _kernel7 = np.concatenate([laplace7 for _ in range(out_channel)], axis=4)
        if in_channel==1:
            kernel7 = _kernel7
        else:
            kernel7 = np.concatenate([_kernel7 for _ in range(in_channel)], axis=3)
        return kernel7
    else:
        assert False, 'Not supported size, must be 3/5/7'


def spatial_gradients(in_channel=3,mode='sobel', name='gaussian_filter'):
    if mode=='sobel':
        y_kernel = np.array(
            [[[[[-1]],[[0]],[[1]]],[[[-2]],[[0]],[[2]]],[[[-1]],[[0]],[[1]]]]], dtype=np.float32) # (D,H,W,in_channel,out_channel)
        x_kernel = np.array(
            [[[[[-1]],[[-2]],[[-1]]],[[[0]],[[0]],[[0]]],[[[1]],[[2]],[[1]]]]], dtype=np.float32)
        if in_channel==1:
            return (y_kernel,x_kernel)
        else:
            g_y = np.concatenate([y_kernel for _ in range(in_channel)], axis=3)
            g_x = np.concatenate([x_kernel for _ in range(in_channel)], axis=3)
        return (g_y,g_x)
    if mode=='roberts':
        y_kernel = np.array(
            [[[[[1]],[[0]]],[[[0]],[[-1]]]]], dtype=np.float32)
        x_kernel = np.array(
            [[[[[0]],[[1]]],[[[-1]],[[0]]]]], dtype=np.float32)
        if in_channel==1:
            return (y_kernel,x_kernel)
        else:
            g_y = np.concatenate([y_kernel for _ in range(in_channel)], axis=3)
            g_x = np.concatenate([x_kernel for _ in range(in_channel)], axis=3)  
     
        return (g_y,g_x) 


def Gaussian_template(theta,accept_field=1,dimensions=2):
    # 均值为0，方差为theta的离散高斯模板，theta越大，曲线越矮胖
    size = []
    [size.append(2*accept_field+1) for _ in range(dimensions)]
    H = np.zeros(shape=size,dtype=np.float32)
    index = [range(0, 2*accept_field+1) for _ in range(dimensions)]
    if len(index) == 3:
        for i in index[0]:
            for j in index[1]:
                for k in index[2]:
                    H[i,j,k] = (1.0/(2*math.pi*theta**2))*math.exp(
                        -(pow(i-accept_field,2)+pow(j-accept_field,2)+pow(k-accept_field,2))/(2*theta**2))

    elif len(index) == 2:
        for i in index[0]:
            for j in index[1]:
                H[i, j] = (1.0 / (2 * math.pi * theta ** 2)) * math.exp(
                    -(pow(i - accept_field, 2) + pow(j - accept_field, 2)) / (2 * theta ** 2))

    else:
        assert False, 'only 2/3-dimension kernel supported now,if ' \
                                                   'you want customize you own kernel,change cycle times bellow'
    return H


if __name__ == '__main__':
    _k = spatial_gradients()
    print(_k[0].shape,_k[1].shape)
    print(_k[0],_k[1])

