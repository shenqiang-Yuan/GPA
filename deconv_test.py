"""
    strides = [1, 3, 3]
    mask = tf.layers.conv3d_transpose(attention, 1,
                                      (1,3,3),
                                      strides,
                                      activation=tf.nn.relu,
                                      use_bias=False,
                                      padding="SAME",
                                      name='mask_level1',
                                      data_format="channels_last")
    strides = [1, 2, 2]
    mask = tf.layers.conv3d_transpose(mask, 1,
                                      (1,3,3),
                                      strides,
                                      activation=tf.nn.relu,
                                      use_bias=False,
                                      padding="SAME",
                                      name='mask_level2',
                                      data_format="channels_last")
    mask = tf.transpose(mask, (0,4,2,3,1),name='pre_crop')
    mask = tf.squeeze(mask, axis=1,name='mask_pre' )

    shape = mask.shape.as_list()
    crop_size = tf.constant([112,112],dtype=tf.int32)
    h_0 = (shape[1]-112)/(2.0*shape[1])
    w_0 = (shape[2]-112)/(2.0*shape[2])
    h_1 = ((shape[1]-112)/2.0 + 112)/shape[1]
    w_1 = ((shape[2]-112)/2.0 + 112)/shape[2]
    boxes = tf.constant([[h_0,w_0,h_1,w_1]]*shape[0],dtype=tf.float32, name='fixed_scale')
    box_ind = tf.constant([i for i in range(shape[0])],dtype=tf.int32)

    final_mask = tf.image.crop_and_resize(mask,
                                          boxes,
                                          box_ind,
                                          crop_size,
                                          name='mask_per_images')

"""
import tensorflow as tf
from dataset import Dataset, Dataloader
from tensorboardX import SummaryWriter
import numpy as np

# [9, 1, 16, 112, 112]
def models(inputs,class_num=101,scope='try_model'):
    with tf.variable_scope(scope):
        mid = tf.squeeze(inputs, axis=-1)

        in_channels = mid.shape.as_list()[1]
        w = tf.get_variable(name='c0_/kernel', shape=(5,5,in_channels,1))
        mid = tf.nn.depthwise_conv2d(mid,w,
                                     strides=(1,1,2,2),
                                     padding='SAME',
                                     rate=(3, 3),
                                     name='depthwise0_group',
                                     data_format='NCHW')
        mid = tf.nn.relu(mid,name='mid_1')
        multiplier = 16
        w1 = tf.get_variable(name='c1_/kernel', shape=(3,3,in_channels,multiplier))
        mid = tf.nn.depthwise_conv2d(mid, w1, strides=(1, 1, 2, 2), padding='VALID',
                                     rate=(1, 1),
                                     name='depthwise1_group',
                                     data_format='NCHW')

        mid = tf.split(mid,multiplier,axis=1,name='splits_for_channel_1')
        mid0 = tf.stack(mid[0:7],axis=4,name='stack_1')  # a new axis for channels
        mid1 = tf.stack(mid[8:15],axis=4,name='stack_2')
        shape = mid0.shape.as_list()
        mid0 = tf.reshape(mid0,(shape[0],shape[1],shape[2]*shape[3],shape[4]),name='prepare_attention_0')
        mid1 = tf.reshape(mid1,(shape[0],shape[1],shape[4],shape[2]*shape[3]),name='prepare_attention_1')
        mid = tf.matmul(mid0,mid1,name='attention_fusion')
        attention = tf.expand_dims(mid,axis=4,name='attention_dims')
        fet = tf.layers.conv3d(attention,16,(3,3,3),
                               strides=(2,2,2),
                               use_bias=False,
                               activation=tf.nn.sigmoid,
                               padding='VALID',
                               data_format='channels_last',
                               name='conv3d_0')
        fet = tf.layers.batch_normalization(fet,axis=4,momentum=0.98,epsilon=1e-5,name='batchnorm_0')
        fet = tf.layers.max_pooling3d(fet,pool_size=(1,3,3),strides=(1,2,2),name='max_pool_0')
        fet = tf.layers.conv3d(fet, 16, (3, 3, 3),
                               strides=(2, 2, 2),
                               use_bias=False,
                               activation=tf.nn.sigmoid,
                               padding='VALID',
                               data_format='channels_last',
                               name='conv3d_1')
        fet = tf.layers.batch_normalization(fet, axis=4, momentum=0.98, epsilon=1e-5, name='batchnorm_1')
        fet = tf.layers.max_pooling3d(fet,pool_size=(1,3,3),strides=(1,2,2),name='max_pool_1')
        fet = tf.layers.conv3d(fet, 32, (3, 3, 3),
                               strides=(2, 2, 2),
                               use_bias=False,
                               activation=tf.nn.sigmoid,
                               padding='VALID',
                               data_format='channels_last',
                               name='conv3d_2')
        fet = tf.layers.batch_normalization(fet, axis=4, momentum=0.98, epsilon=1e-5, name='batchnorm_2')
        fet = tf.layers.max_pooling3d(fet, pool_size=(1, 3, 3), strides=(1, 2, 2), name='max_pool_2')
        fet = tf.layers.conv3d(fet, 64, (1, 2, 2),
                               strides=(1, 1, 1),
                               use_bias=False,
                               activation=tf.nn.sigmoid,
                               padding='VALID',
                               data_format='channels_last',
                               name='conv3d_3')
        fet = tf.layers.batch_normalization(fet, axis=4, momentum=0.98, epsilon=1e-5, name='batchnorm_3')

        fet = tf.reshape(fet,(shape[0],-1),name='flatten')
        fet = tf.layers.dense(fet,class_num,activation=tf.nn.sigmoid,use_bias=True,name='L%d' % class_num)
    return fet


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 9, 'batch_size to specify the input_placeholder')
tf.flags.DEFINE_boolean('checkpoint', False, "is there snapshot to load to continue training")
tf.flags.DEFINE_boolean('pretrained', False, 'use pretrained models for hot starting')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    _SAMPLE_VIDEO_FRAMES = 16
    _IMAGE_SIZE = 112
    NUM_CLASS = 101
    _CHECKPOINT_PATHS = {
        'pretrained_model': 'pretrained-models/r2plus1-18/Caffe2TfR2.5d.ckpt',
        'snapshots': 'saved_models/model.ckpt'
    }
    is_checkpoint = FLAGS.checkpoint
    batch_size = FLAGS.batch_size
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

    graph = tf.get_default_graph()
    with graph.as_default():
        data = tf.placeholder(shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 1), dtype=tf.float32)
        label = tf.placeholder(shape=(batch_size, NUM_CLASS), dtype=tf.int32)
        result = models(data,
                        class_num=101,
                        scope='at_model')
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        # global_step = tf.get_variable("global_step", shape=(),trainable=False,initializer=tf.constant_initializer(0), dtype=tf.int64)

        variable_map = {}
        regular_map = {}
        extra_map = {}
        # Note: key of variable_map should be specific enough to match the unique variable
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var.name, var.shape)
            if 'L101' in var.name:
                extra_map[var.name.replace(":0", '')] = var
                continue
            elif 'kernel:0' in var.name:
                regular_map[var.name.replace(":0", '')] = var
                continue
            else:
                variable_map[var.name.replace(":0", '')] = var

        # rgb_loader = tf.train.Saver(var_list=dict(variable_map,**regular_map), reshape=True)
        rgb_resaver = tf.train.Saver(var_list=dict(variable_map, **regular_map, **extra_map), reshape=True)
        """
        with tf.Session() as sess:
            global_init = tf.global_variables_initializer()
            sess.run(global_init)
            rgb_loader.restore(sess, _CHECKPOINT_PATHS['pretrained_model'])
            rgb_resaver.save(sess, _CHECKPOINT_PATHS['snapshots'])
        """

        top1_prediction = tf.math.in_top_k(result, tf.argmax(label, 1), 1)
        top5_prediction = tf.math.in_top_k(result, tf.argmax(label, 1), 5)
        acc_top1 = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
        acc_top5 = tf.reduce_mean(tf.cast(top5_prediction, tf.float32))
        weight_deacy_loss = tf.reduce_mean([0.005 * tf.nn.l2_loss(var) for var in list(regular_map.values())[-6:]])
        # weight_deacy_loss = 0.0
        # entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=label), axis=0)
        entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=label), axis=0)
        learning_rate = tf.train.exponential_decay(learning_rate=0.05, global_step=global_step, decay_steps=1200,
                                                   decay_rate=0.98)
        loss = entropy_loss + weight_deacy_loss
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        # train_step = opt.minimize(loss, global_step=global_step)

        grads_var = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grads_var, global_step=global_step)
        with tf.name_scope("train_step"):
            tf.summary.scalar("accuracy_top1", acc_top1)
            tf.summary.scalar("accuracy_top5", acc_top5)
            tf.summary.scalar("entropy_loss", loss)
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
            #sess.run(tf.assign(global_step, 13767))
            epoch = 0
            while epoch <= 6:
                for i, (datas, labells) in enumerate(train_generator):
                    if i > num_train - 1:
                        break
                    feed_dict[data] = datas[:, :, :, :, np.newaxis]
                    feed_dict[label] = labells
                    _, Entropyloss, top1, top5, summaries = sess.run([train_op, loss, acc_top1, acc_top5, merge1],
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
                    _Entropyloss, _top1, _top5 = sess.run([loss, acc_top1, acc_top5],
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