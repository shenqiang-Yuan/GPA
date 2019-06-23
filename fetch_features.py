import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import pickle
import os

from laplace_temporal_net import Dataloader,Dataset,create_r3d
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpus=[0]
batchsize_gpu_0 = 8
# batchsize_gpu_1 = 5
# assignments = [batchsize_gpu_0,batchsize_gpu_1]
assignments = [batchsize_gpu_0]

checkpoint_path = 'saved_models/checkpoint/model.ckpt'

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    _SAMPLE_VIDEO_FRAMES = 32
    _IMAGE_SIZE = 112
    NUM_CLASS = 101
    batch_size = 0
    for n in assignments:
        batch_size += n
    assert batch_size >= 1
    train_set = Dataset.DataSet(clip_length=_SAMPLE_VIDEO_FRAMES,
                                sample_step=2,
                                data_root='/home/pr606/Pictures/ucf_images',
                                annotation_path='/home/pr606/Pictures/ucf101_json_file/ucf101_01.json',
                                spatial_transform=None,
                                mode='train')
    validate_set = Dataset.DataSet(clip_length=_SAMPLE_VIDEO_FRAMES,
                                   sample_step=2,
                                   data_root='/home/pr606/Pictures/ucf_images',
                                   annotation_path='/home/pr606/Pictures/ucf101_json_file/ucf101_01.json',
                                   spatial_transform=None,
                                   mode='validation')
    train_generator = Dataloader.DataGenerator(train_set, batch_size=batch_size, shuffle=False)
    validate_generator = Dataloader.DataGenerator(validate_set, batch_size=batch_size, shuffle=False)
    num_train = train_generator.__len__()
    num_validate = validate_generator.__len__()

    print("training data num is %d" % num_train)  # 733
    print("validation data num is %d" % num_validate)  # 291

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
        if '_com2_GPA' in var.name:
            features.append(var)
            continue
        # parall_model_1/final_avg/AvgPool3D:0
        if '_com4_GPA' in var.name:
            features.append(var)
            continue
    assert len(features) != 0
    final_results = results[0] if len(results) == 1 else tf.concat(results, axis=0, name='merge_result_across_gpus')
    final_features = features[0] if len(features) == 1 else tf.concat(features, axis=0, name='merge_features_across_gpus')

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

        all_activations = {'final': [],'mapping': []}
        for i, (datas, labells,video_names) in enumerate(train_generator):
            if i > num_train - 1:
                break
            feed_dict[data] = datas#[:, :, :, :, np.newaxis]
            feed_dict[label] = labells
            feat,acc1,acc5 = sess.run([final_features,acc_top1, acc_top5],feed_dict=feed_dict)
            all_activations['final'].append(feat)
            all_activations['mapping'].append(video_names)
            print('acc1: {:.6f}, acc5: {:.6f}, fetch features with shape of :{},remaining {} samples'
                  .format(acc1, acc5, feat.shape, (num_train-1-i)*batch_size))

    for key in all_activations:
        all_activations[key] = np.concatenate(all_activations[key],axis=0)
    print(all_activations['final'].shape)
    with open('resnet-GPA-10-features.pickle', 'wb') as handle:
        pickle.dump(all_activations, handle)


