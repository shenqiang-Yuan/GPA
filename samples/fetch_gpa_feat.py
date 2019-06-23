import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import pickle
import os,json
from samples import read_clips

from laplace_temporal_net import create_r3d
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpus=[0]

video_json_path = 'three_samples.json'
data_root = '/home/pr606/Pictures/ucf_images'
checkpoint_path = '/home/pr606/YUAN/history/tf-R2plus1D/saved_models/model.ckpt'

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    _SAMPLE_VIDEO_FRAMES = 32
    _IMAGE_SIZE = 224
    NUM_CLASS = 101

    with open(video_json_path) as f1:
        annotation_json = json.load(f1)
    data_json = annotation_json["database"]
    label_list = annotation_json["labels"]

    video_name = 'v_GolfSwing_g04_c03'
    video_length = data_json['v_GolfSwing_g04_c03']['annotations']['n_frames']
    labe = data_json['v_GolfSwing_g04_c03']['annotations']['label']
    batchsize_gpu_0 = video_length // _SAMPLE_VIDEO_FRAMES
    frame_indexs = range(video_length)
    video_path = os.path.join(data_root, labe, video_name)

    assignments = [batchsize_gpu_0]
    batch_size = 0
    for n in assignments:
        batch_size += n
    assert batch_size >= 1
    pathes = [video_path]*batchsize_gpu_0

    one_batch = read_clips.get_batch(pathes,[frame_indexs[i*_SAMPLE_VIDEO_FRAMES:(i+1)*_SAMPLE_VIDEO_FRAMES] for i in range(batchsize_gpu_0)])

    data = tf.placeholder(shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=tf.float32,name='clips')
    label = tf.placeholder(shape=(batch_size, NUM_CLASS), dtype=tf.int32,name='labels')

    data_list = tf.split(data,assignments,axis=0,name='sep_0_for_%d_gpus'%len(gpus))
    label_list = tf.split(label,assignments,axis=0,name='sep_1_for_%d_gpus'%len(gpus))

    variable_map = {}
    regular_map = {}
    extra_map = {}
    results = []
    for i in gpus:
        with tf.device("/gpu:%d" % i):
            with tf.variable_scope('parall_model',reuse=tf.AUTO_REUSE):
                result = create_r3d(
                    data=data_list[i],
                    model_depth=18,
                    num_labels=NUM_CLASS,
                    num_input_channels=1,
                    is_decomposed=False,
                    no_bias=1,
                    is_test=1,
                    spatial_bn_mom=0.9,
                    final_spatial_kernel=14,
                    final_temporal_kernel=4,
                )
                if i == 0:
                    # Note: key of variable_map should be specific enough to match the unique variable
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        print(var.name, var.shape)
                        if 'L101' in var.name:
                            extra_map[var.name[13:].replace(":0", '')] = var
                            continue
                        elif 'conv1_middle/kernel' in var.name or 'conv1/kernel' in var.name:
                            extra_map[var.name[13:].replace(":0", '')] = var
                            continue
                        elif 'conv1_middle_spatbn_relu' in var.name:
                            extra_map[var.name[13:].replace(":0", '')] = var
                            continue
                        elif 'Lp_conv_' in var.name or 'global_step' in var.name:
                            continue
                        elif 'gaussian_filter' in var.name:
                            continue
                        elif 'kernel:0' in var.name:
                            regular_map[var.name[13:].replace(":0", '')] = var
                            continue
                        else:
                            variable_map[var.name[13:].replace(":0", '')] = var
                        labels = tf.stop_gradient(label_list[i],
                                                  name='disallow_grad_labels%d' % i)
                        top1_prediction = tf.nn.in_top_k(result, tf.argmax(labels, 1), 1)
                        top5_prediction = tf.nn.in_top_k(result, tf.argmax(labels, 1), 5)
                else:
                    labels = tf.stop_gradient(label_list[i], name='disallow_grad_labels%d' % i)
                    top1_prediction = tf.concat([top1_prediction, tf.nn.in_top_k(result, tf.argmax(labels, 1), 1)],
                                                axis=0)
                    top5_prediction = tf.concat([top5_prediction, tf.nn.in_top_k(result, tf.argmax(labels, 1), 5)],
                                                axis=0)
                    pass
                tf.get_variable_scope().reuse_variables()
                results.append(result)
                # vs.get_variable_scope().reuse_variables()
    acc_top1 = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
    acc_top5 = tf.reduce_mean(tf.cast(top5_prediction, tf.float32))
    rgb_resaver = tf.train.Saver(var_list=dict(variable_map, **regular_map, **extra_map), reshape=True)
    nodes = tf.get_collection('source_node')
    features = []
    for var in nodes:
        # parall_model/final_avg/AvgPool3D:0
        # features.append(var)
        if '_com2_GPA' in var.name:
            com2_gpa = var
            continue
        # parall_model_1/final_avg/AvgPool3D:0
        # features.append(var)
        if '_com4_GPA' in var.name:
            com4_gpa = var
            continue
    # assert len(features) != 0
    final_results = results[0] if len(results) == 1 else tf.concat(results, axis=0, name='merge_result_across_gpus')
    # final_features = features[0] if len(features) == 1 else tf.concat(features, axis=0, name='merge_features_across_gpus')

    global_init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    with tf.Session(config=config) as sess:
        sess.run(global_init)
        feed_dict = {}
        if os.path.exists(checkpoint_path+'.index'):
            rgb_resaver.restore(sess, checkpoint_path)
            print('restoring checkpoint from {} '.format(checkpoint_path))
            tf.logging.info('RGB checkpoint restored')
        else:
            assert False,'can not find any checkpoint files'
        all_activations = {'com2_gpa': None, 'com4_gpa': None, 'mapping': None}
        feed_dict[data] = one_batch  #[:, :, :, :, np.newaxis]
        feat2,feat4 = sess.run([com2_gpa,com4_gpa], feed_dict=feed_dict)

        all_activations['com2_gpa'] = feat2
        all_activations['com4_gpa'] = feat4

        all_activations['mapping'] = video_path

    print(all_activations['com2_gpa'].shape)
    print(all_activations['com4_gpa'].shape)
    with open(video_name+'.pickle', 'wb') as handle:
        pickle.dump(all_activations, handle)


