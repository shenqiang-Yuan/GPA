import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
from laplace_temporal_net import Dataloader,Dataset,create_r3d,SummaryWriter


def averge(tower_grads):
    averg_grads_vars = []
    for var_and_grad in zip(*tower_grads):
        grads = []
        for g, _ in var_and_grad:
            grad = tf.expand_dims(g, 0)
            grads.append(grad)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = var_and_grad[0][1]
        grad_var = (grad, v)
        averg_grads_vars.append(grad_var)
    return averg_grads_vars


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 9, 'batch_size to specify the input_placeholder')
tf.flags.DEFINE_boolean('checkpoint', False, "is there snapshot to load to continue training")
tf.flags.DEFINE_boolean('pretrained', False, 'use pretrained models for hot starting')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    _SAMPLE_VIDEO_FRAMES = 16
    _IMAGE_SIZE = 112
    NUM_CLASS = 101
    gpus=[0]
    _CHECKPOINT_PATHS = {
        'pretrained_model': 'pretrained-models/r2plus1-18/Caffe2TfR2.5d.ckpt',
        'snapshots': 'saved_models/model.ckpt'
    }
    is_checkpoint = FLAGS.checkpoint
    batch_size = FLAGS.batch_size * len(gpus)
    use_pretrained = FLAGS.pretrained

    train_set = Dataset.DataSet(clip_length=_SAMPLE_VIDEO_FRAMES,
                                sample_step=2,
                                data_root='/home/pr606/Pictures/UCF101DATASET/ucf101',
                                annotation_path='/home/pr606/Pictures/dataset_annotations/ucf101_json_file/ucf101_01.json',
                                spatial_transform=None,
                                mode='train')
    validate_set = Dataset.DataSet(clip_length=_SAMPLE_VIDEO_FRAMES,
                                   sample_step=2,
                                   data_root='/home/pr606/Pictures/UCF101DATASET/ucf101',
                                   annotation_path='/home/pr606/Pictures/dataset_annotations/ucf101_json_file/ucf101_01.json',
                                   spatial_transform=None,
                                   mode='validation')
    train_generator = Dataloader.DataGenerator(train_set, batch_size=batch_size)
    validate_generator = Dataloader.DataGenerator(validate_set, batch_size=batch_size)
    num_train = train_generator.__len__()
    num_validate = validate_generator.__len__()

    print("training data num is %d" % num_train)  # 733
    print("validation data num is %d" % num_validate)  # 291

    data = tf.placeholder(shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 1), dtype=tf.float32,name='clips')
    label = tf.placeholder(shape=(batch_size, NUM_CLASS), dtype=tf.int32,name='labels')
    data_list = tf.split(data,len(gpus),axis=0,name='sep_0_for_%d'%len(gpus))
    label_list = tf.split(label,len(gpus),axis=0,name='sep_1_for_%d'%len(gpus))
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    # global_step = tf.get_variable("global_step", shape=(),trainable=False,initializer=tf.constant_initializer(0), dtype=tf.int64)
    graph = tf.get_default_graph()
    learning_rate = tf.train.exponential_decay(learning_rate=0.005, global_step=global_step, decay_steps=1200,
                                               decay_rate=0.98)
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    # train_step = opt.minimize(loss, global_step=global_step)

    grads_vars = []
    variable_map = {}
    regular_map = {}
    extra_map = {}
    for i in gpus:
        with tf.device("/gpu:%d" % i):
            # tf.name_scope()不会为变量自动添加前缀，
            # 可以保证重用的变量是之前唯一创建的,这与tf.variable_scope()不同
            with tf.name_scope('model_%d' % i):
                result = create_r3d(
                    data=data_list[i],
                    model_depth=10,
                    num_labels=NUM_CLASS,
                    num_input_channels=1,
                    is_decomposed=True,
                    no_bias=1,
                    spatial_bn_mom=0.9,
                    final_temporal_kernel=2,
                )

                if i == 0:
                    labels = tf.stop_gradient(label_list[i],
                                              name='disallow_grad_labels%d' % i)

                    entropy_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=labels),
                        axis=0)

                    top1_prediction = tf.math.in_top_k(result, tf.argmax(labels, 1), 1)
                    top5_prediction = tf.math.in_top_k(result, tf.argmax(labels, 1), 5)
                    acc_top1 = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
                    acc_top5 = tf.reduce_mean(tf.cast(top5_prediction, tf.float32))
                    grad_var = opt.compute_gradients(entropy_loss,
                                                     var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope='parall_model'))
                    grads_vars.append(grad_var)

                    tf.get_variable_scope().reuse_variables()

                    # Note: key of variable_map should be specific enough to match the unique variable
                    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        print(var.name, var.shape)
                        if 'L101' in var.name:
                            extra_map[var.name.replace(":0", '')] = var
                            continue
                        elif 'conv1_middle/kernel' in var.name or 'conv1/kernel' in var.name:
                            extra_map[var.name.replace(":0", '')] = var
                            continue
                        elif 'conv1_middle_spatbn_relu' in var.name:
                            extra_map[var.name.replace(":0", '')] = var
                            continue
                        elif 'Lp_conv_' in var.name or 'global_step' in var.name:
                            continue
                        elif 'gaussian_filter' in var.name:
                            continue
                        elif 'kernel:0' in var.name:
                            regular_map[var.name.replace(":0", '')] = var
                            continue
                        else:
                            variable_map[var.name.replace(":0", '')] = var
                    # weight_deacy_loss = 0.0
                    weight_deacy_loss = tf.reduce_mean(
                            [0.005 * tf.nn.l2_loss(var) for var in list(regular_map.values())[-6:]])

                else:
                    labels = tf.stop_gradient(label_list[i], name='disallow_grad_labels%d' % i)

                    loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=labels), axis=0)
                    grad_var = opt.compute_gradients(loss,
                                                     var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope='parall_model'))
                    grads_vars.append(grad_var)
    averge_grads_vars = averge(grads_vars)
    # rgb_loader = tf.train.Saver(var_list=dict(variable_map,**regular_map), reshape=True)
    rgb_resaver = tf.train.Saver(var_list=dict(variable_map, **regular_map, **extra_map), reshape=True)
    """
    with tf.Session() as sess:
        global_init = tf.global_variables_initializer()
        sess.run(global_init)
        rgb_loader.restore(sess, _CHECKPOINT_PATHS['pretrained_model'])
        rgb_resaver.save(sess, _CHECKPOINT_PATHS['snapshots'])
    """

    train_step = opt.apply_gradients(averge_grads_vars, global_step=global_step)
    with tf.name_scope("train_step"):
        tf.summary.scalar("accuracy_top1", acc_top1)
        tf.summary.scalar("accuracy_top5", acc_top5)
        tf.summary.scalar("entropy_loss", entropy_loss)
    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([train_step, update_ops]):
        train_op = tf.no_op(name='train')

    merge1 = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="train_step"))
    train_writer = tf.summary.FileWriter("./logs/train", graph)
    validate_writer = SummaryWriter("./logs/validate", graph)  # tensorboardX

    global_init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    with tf.Session(config=config) as sess:
        sess.run(global_init)
        feed_dict = {}
        if is_checkpoint:
            rgb_resaver.restore(sess, _CHECKPOINT_PATHS['snapshots'])
            tf.logging.info('RGB checkpoint restored')
        '''according to the value of global_step of saved graph if necessary:sess.run(tf.assign(global_step, 500))'''
        # sess.run(tf.assign(global_step, 13767))
        epoch = 0
        while epoch <= 8:
            for i, (datas, labells) in enumerate(train_generator):
                if i > num_train - 1:
                    break
                feed_dict[data] = datas[:, :, :, :, np.newaxis]
                feed_dict[label] = labells
                _, Entropyloss, top1, top5, summaries = sess.run([train_op, entropy_loss, acc_top1, acc_top5, merge1],
                                                                 feed_dict=feed_dict)
                step = global_step.eval()
                train_writer.add_summary(summaries, step)
                print('steps/epochs:{}/{}===> train_loss: {:.6f}, acc_top1: {:.4f}, acc_top5: {:.4f}'
                      .format(step, epoch, Entropyloss, top1, top5))
            top1 = 0.0
            top5 = 0.0
            Entropyloss = 0.0
            k = 0
            for i, (datas, labells) in enumerate(validate_generator):

                feed_dict[data] = datas[:, :, :, :, np.newaxis]
                feed_dict[label] = labells
                _Entropyloss, _top1, _top5 = sess.run([entropy_loss, acc_top1, acc_top5],
                                                      feed_dict=feed_dict)
                Entropyloss += _Entropyloss
                top1 += _top1
                top5 += _top5
                k += 1
                print("steps/epochs:{}/{}===>validate_loss:{:.6f},acc_top1:{:.4f},acc_top5: {:.4f}"
                      .format(k, epoch, _Entropyloss, _top1, _top5))
                if i >= num_validate - 1:
                    break
                if k >= 300:
                    break
            top1 = float(top1) / k
            top5 = float(top5) / k
            Entropyloss = float(Entropyloss) / k
            validate_writer.add_scalar("validate/mean_loss", Entropyloss, epoch)
            validate_writer.add_scalar("validate/mean_accuracy_top1", top1, epoch)
            validate_writer.add_scalar("validate/mean_accuracy_top5", top5, epoch)
            epoch += 1

            rgb_resaver.save(sess, _CHECKPOINT_PATHS['snapshots'])
        train_writer.close()
        validate_writer.close()
