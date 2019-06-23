from attention_models import VideoModelBuilder
from pre_process import tf, Gaussian, Batch_Norm, get_temporal_kernel,spatial_gradients
from dataset import Dataset, Dataloader
from tensorboardX import SummaryWriter
import numpy as np
import pickle
import pandas as pd

DEPTH_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 3, 4, 3),
    34: (3, 4, 6, 3),
}

def Laplace_Conv(inputs,in_channel,out_channel,ksize,name='Laplace_conv'):
    _kernel = get_temporal_kernel(ksize, in_channel, out_channel)
    kernel = tf.get_variable(name=name + str(ksize), shape=(ksize, 1, 1, in_channel, out_channel), dtype=tf.float32,
                                initializer=tf.constant_initializer(_kernel))
    return tf.nn.conv3d(inputs, kernel, (1, 1, 1, 1, 1), padding='SAME')


def gradients_xy(inputs,in_channel=1,mode='roberts'):
    rob_y,rob_x = spatial_gradients(in_channel=in_channel,mode=mode)
    #rob_yx = tf.concat([rob_y,rob_x], axis=-1)
    grident_y = tf.nn.conv3d(inputs,rob_y,strides=(1, 1, 1, 1, 1), padding='SAME',dilations=(1,1,1,1,1))
    grident_x = tf.nn.conv3d(inputs,rob_x,strides=(1, 1, 1, 1, 1), padding='SAME',dilations=(1,1,1,1,1))

    return grident_y,grident_x


def create_r3d(
        data,
        num_labels,
        num_input_channels=3,
        label=None,
        is_test=0,
        logits=0,
        no_bias=1,
        final_spatial_kernel=7,
        final_temporal_kernel=1,
        model_depth=18,
        is_decomposed=False,
        spatial_bn_mom=0.9,
):
    data = Batch_Norm(data,'norm')
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
        '''
        data = Gaussian(data, in_channels=1,dimensions=2, name='gaussian_filter')
        manual_features1 = Laplace_Conv(data, ksize=3, in_channel=1, out_channel=1, name='Lp_conv_')
        manual_features2 = Laplace_Conv(data, ksize=5, in_channel=1, out_channel=1, name='Lp_conv_')
        manual_features3 = Laplace_Conv(data, ksize=7, in_channel=1, out_channel=1, name='Lp_conv_')
        g_y,g_x = gradients_xy(data,in_channel=1)
        net = tf.concat([net,manual_features1,manual_features2,manual_features3,g_y,g_x], axis=-1)
        '''
        #handCraft = [manual_features1,manual_features2,manual_features3,g_y,g_x]
        #squeezed = lambda ten:tf.squeeze(ten, axis=-1)
        #stacking_on_temp = [squeezed(f) for f in handCraft]
        #NHWC = lambda ten:tf.transpose(ten,(0,2,3,1))
        #groups = [NHWC(f) for f in stacking_on_temp]
        #for g in range(len(stacking_on_temp)):
        #    pass

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
        net = tf.nn.relu(net, name='conv1_middle_activ_relu')
        net = tf.layers.conv3d(net,
                               filters=64,
                               kernel_size=[3, 1, 1],
                               strides=[1, 1, 1],
                               padding='same',
                               use_bias=not no_bias,
                               kernel_initializer=tf.constant_initializer(get_temporal_kernel(3,45,64)),
                               name='conv1')
    net = tf.layers.batch_normalization(net, training=not is_test,
                                        momentum=spatial_bn_mom,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        name='conv1_spatbn_relu')
    net = tf.nn.relu(net, name='conv1_spatbn_activ')
    tf.add_to_collection(source, net)
    (n1, n2, n3, n4) = DEPTH_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(use_bias=not no_bias,
                                is_test=is_test,
                                spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        net = builder.add_simple_block(net, 64, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)

    # conv_3x
    net = builder.add_simple_block(
        net, 128, down_sampling=True, is_decomposed=is_decomposed)
    # net = builder.GPA(net,mid_channels=64,index=0,v_nodes=128,reduce_factor=9)

    tf.add_to_collection(source, net)

    for _ in range(n2 - 1):
        net = builder.add_simple_block(net, 128, is_decomposed=is_decomposed)
    tf.add_to_collection(source, net)
    # conv_4x
    net = builder.add_simple_block(
        net, 256, down_sampling=True, is_decomposed=is_decomposed)
    # net = builder.GPA(net, mid_channels=64, index=0, v_nodes=64, reduce_factor=7)

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
    net = tf.layers.dropout(net, rate=0.4,name='dropout')
    last_out = tf.layers.dense(net, num_labels, use_bias=True, name='last_out_L{}'.format(num_labels))
    tf.add_to_collection(source, last_out)

    if logits:
        # For features extraction, we return logits
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_out, labels=label), axis=0)
        return (last_out, loss)
    else:
        # For inference, we just return softmax
        softmax = tf.nn.softmax(last_out, name='softmax')
        return softmax


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 11, 'batch_size to specify the input_placeholder')
tf.flags.DEFINE_boolean('checkpoint', True, "is there snapshot to load to continue training")
tf.flags.DEFINE_boolean('pretrained', False, 'use pretrained models for hot starting')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    _SAMPLE_VIDEO_FRAMES = 16
    _IMAGE_SIZE = 112
    NUM_CLASS = 101
    _CHECKPOINT_PATHS = {
        'pretrained_model': 'pretrained-models/r3d-18/resnet-18-l16.ckpt',
        'snapshots': 'saved_models/model.ckpt',
        'logs_train': './logs/train',
        'logs_val': './logs/validate',
    }
    is_checkpoint = FLAGS.checkpoint
    batch_size = FLAGS.batch_size
    use_pretrained = FLAGS.pretrained

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
    train_generator = Dataloader.DataGenerator(train_set, batch_size=batch_size)
    validate_generator = Dataloader.DataGenerator(validate_set, batch_size=batch_size)
    num_train = train_generator.__len__()
    num_validate = validate_generator.__len__()

    print("training data num is %d" % num_train)    # 733
    print("validation data num is %d" % num_validate)    # 291
    
    graph = tf.get_default_graph()
    with graph.as_default():
        data = tf.placeholder(shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=tf.float32)
        label = tf.placeholder(shape=(None, NUM_CLASS), dtype=tf.int32)
        lab = tf.stop_gradient(label, name='disallow_grad')
        result = create_r3d(
            data=data,
            model_depth=18,
            num_labels=NUM_CLASS,
            num_input_channels=1,
            is_decomposed=False,
            no_bias=1,
            logits=1,
            is_test=0,
            spatial_bn_mom=0.9,
            final_temporal_kernel=2,
        )
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        # global_step = tf.get_variable("global_step", shape=(),trainable=False,initializer=tf.constant_initializer(0), dtype=tf.int64)

        variable_map = {}
        regular_map = {}
        extra_map = {}
        # Note: key of variable_map should be specific enough to match the unique variable
        i = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var.name, var.shape,i+1)
            i += 1
            if 'global_step' in var.name:
                continue
            elif 'L101' in var.name:
                extra_map[var.name.replace(":0", '')] = var
                continue
            elif "GPA" in var.name:
                extra_map[var.name.replace(":0", '')] = var
                continue
            elif 'kernel:0' in var.name:
                regular_map[var.name.replace(":0", '')] = var
                continue
            else:
                variable_map[var.name.replace(":0", '')] = var

        # rgb_loader = tf.train.Saver(var_list=dict(variable_map, **regular_map), reshape=True)
        rgb_resaver = tf.train.Saver(var_list=dict(variable_map, **regular_map, **extra_map), reshape=True)
        '''
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            global_init = tf.global_variables_initializer()
            sess.run(global_init)
            rgb_loader.restore(sess, _CHECKPOINT_PATHS['pretrained_model'])
            rgb_resaver.save(sess, _CHECKPOINT_PATHS['snapshots'])
        '''
        pass
        """
        with open('/home/pr606/YUAN/torch_project/pklfiles/resnet3d_34_kinetics.pkl', 'rb') as fopen:
            blobs = pickle.load(fopen, encoding='latin1')
        data = pd.read_csv('/home/pr606/YUAN/torch_project/map.csv',header=None,delimiter=';')
        new_tensor = data[0].values
        orignal_tensor = data[1].values
        shapes = data[2].values
        with tf.Session() as sess:
            global_init = tf.global_variables_initializer()
            sess.run(global_init)
            i = 1
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if var.name in new_tensor:
                    id = np.argwhere(new_tensor==var.name)[0,0]
                    para = blobs[orignal_tensor[id]]
                    if len(eval(shapes[id]))==5:
                        para = para.transpose(2,3,4,1,0)
                    elif len(eval(shapes[id]))==1:
                        pass
                    elif len(eval(shapes[id]))==2:
                        pass
                    else:
                        pass
                    sess.run(tf.assign(var, para))
                    print(i)
                    i +=1

            #rgb_loader.restore(sess, _CHECKPOINT_PATHS['pretrained_model'])
            #rgb_resaver.save(sess, _CHECKPOINT_PATHS['snapshots'])
            rgb_resaver.save(sess, '/home/pr606/YUAN/history/tf-R2plus1D/pretrained-models/r3d-34/model.ckpt')
        
        """
        top1_prediction = tf.nn.in_top_k(result, tf.argmax(lab, 1), 1)
        top5_prediction = tf.nn.in_top_k(result, tf.argmax(lab, 1), 5)
        acc_top1 = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
        acc_top5 = tf.reduce_mean(tf.cast(top5_prediction, tf.float32))
        weight_deacy_loss = tf.reduce_mean([0.005*tf.nn.l2_loss(var) for var in list(regular_map.values())])
        # weight_deacy_loss = 0.0
        # entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=label), axis=0)
        entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=lab), axis=0)

        loss = entropy_loss + weight_deacy_loss
        # learning_rate = tf.train.exponential_decay(learning_rate=0.005, global_step=global_step, decay_steps=1200,
        #                                            decay_rate=0.98)
        init_lr_rate = 0.1
        learning_rate = tf.Variable(name='learn_rate', initial_value=init_lr_rate, dtype=tf.float32, trainable=False)
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # opt = tf.train.AdamOptimizer(5e-3)
        # train_step = opt.minimize(loss, global_step=global_step)

        grads_var = opt.compute_gradients(loss)
        train_step = opt.apply_gradients(grads_var, global_step=global_step)
        with tf.name_scope("summaries_train_and_validation"):
            tf.summary.scalar("accuracy_top1", acc_top1)
            tf.summary.scalar("accuracy_top5", acc_top5)
            tf.summary.scalar("entropy_loss", loss)
        update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies([train_step, update_ops]):
            train_op = tf.no_op(name='train')

        merge1 = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="summaries_train_and_validation"))
        train_writer = tf.summary.FileWriter(_CHECKPOINT_PATHS["logs_train"], graph)
        validate_writer = tf.summary.FileWriter(_CHECKPOINT_PATHS["logs_val"], graph)

        global_init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = 0.85
        with tf.Session(config=config) as sess:
            sess.run(global_init)
            feed_dict = {}
            if is_checkpoint:
                rgb_resaver.restore(sess, _CHECKPOINT_PATHS['snapshots'])
                tf.logging.info('RGB checkpoint restored')
            '''according to the value of global_step of saved graph if necessary:sess.run(tf.assign(global_step, 500))'''
            sess.run(tf.assign(global_step, 22543))
            validation_steps = 7800
            epoch = 26
            while epoch <= 33:
                if epoch%16 == 0 and epoch != 0:
                    init_lr_rate = init_lr_rate/10
                    sess.run(tf.assign(learning_rate, init_lr_rate))
                for i, (datas, labells) in enumerate(train_generator):
                    feed_dict[data] = datas             #[:,:,:,:,np.newaxis]
                    feed_dict[label] = labells
                    _, Entropyloss, top1, top5, summaries = sess.run([train_op, loss, acc_top1, acc_top5, merge1],
                                                                     feed_dict=feed_dict)
                    step = global_step.eval()
                    train_writer.add_summary(summaries, step)
                    print('steps/epochs:{}/{}===> train_loss: {:.6f}, acc_top1: {:.4f}, acc_top5: {:.4f}'
                          .format(step, epoch, Entropyloss, top1, top5))
                    if i >= num_train - 1:
                        break
                    """"""
                k = 0
                for j, (datas, labells) in enumerate(validate_generator):
                    feed_dict[data] = datas             #[:,:,:,:,np.newaxis]
                    feed_dict[label] = labells
                    _Entropyloss, _top1, _top5, summaries = sess.run([loss, acc_top1, acc_top5, merge1],
                                                          feed_dict=feed_dict)
                    k += 1
                    validation_steps +=1
                    print("steps/epochs:{}/{}===>validate_loss:{:.6f},acc_top1:{:.4f},acc_top5: {:.4f}"
                          .format(k, epoch, _Entropyloss, _top1, _top5))
                    validate_writer.add_summary(summaries, validation_steps)
                    
                    if j >= num_validate - 1 or j >= 300:
                        break
                epoch += 1

                rgb_resaver.save(sess, _CHECKPOINT_PATHS['snapshots'])
                
            train_writer.close()
            validate_writer.close()
        
        

