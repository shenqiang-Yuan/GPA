from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
import tensorflow as tf
import numpy as np
from model_builder import build_model
from dataset import Dataset, Dataloader
import pickle
batch_size = 1
_SAMPLE_VIDEO_FRAMES = 24
_LABEL_MAP_PATH = '/home/pr606/python_vir/yuan/i3d-kinects/data/label_map.txt'
with open(_LABEL_MAP_PATH) as f2:
    kinetics_classes = [x.strip() for x in f2.readlines()]

validate_set = Dataset.DataSet(clip_length=_SAMPLE_VIDEO_FRAMES,
                                        sample_step=2,
                                        data_root='/home/pr606/Pictures/part_validate_kinetics',
                                        annotation_path='/home/pr606/python_vir/yuan/EXTRA_DATA/kinetics_part.json',
                                        spatial_transform=None,
                                        mode='validation',
                                        with_start=True,
                                        multi_sample=True
                                        )

validate_generator = Dataloader.DataGenerator(validate_set, batch_size=batch_size, ordered_file_path='/home/pr606/python_vir/yuan/EXTRA_DATA/names_in_order.csv')


num_validate = validate_generator.__len__() # 1005
print("total validate data is :{}".format(num_validate))


inputs = tf.placeholder(shape=(batch_size,_SAMPLE_VIDEO_FRAMES,112,112,3),dtype=tf.float32)

mean, variance = tf.nn.moments(inputs, axes=(0, 1, 2, 3), keep_dims=True, name="normalize_moments")

Gamma = tf.constant(1.0, name="scale_factor", shape=mean.shape, dtype=tf.float32)
Beta = tf.constant(0.0,name="offset_factor", shape=mean.shape, dtype=tf.float32)
data = tf.nn.batch_normalization(inputs, mean, variance, offset=Beta, scale=Gamma, variance_epsilon=1e-3)

'''
normilizing is must at first
'''

result = build_model(data=data,
                         labels=None,
                         model_name='r2plus1d',
                         model_depth=18,
                         num_labels=400,
                         num_channels=3,
                         crop_size=112,
                         clip_length=24,
                         loss_scale=1.0,
                         is_test = 1,
                         )

rgb_saver = tf.train.Saver()
global_init = tf.global_variables_initializer()
feed_dict={}
all_activations = {'final': []}
nodes = tf.get_collection('source_node')
feature = None
for var in nodes:
    if 'final_avg/AvgPool3D:0' in var.name:
        feature = var
        break
assert feature != None

with tf.Session() as sess:
    sess.run(global_init)
    rgb_saver.restore(sess, '/home/pr606/python_vir/yuan/tf-R2plus1D/pretrained-models/r2plus1-18/Caffe2TfR2.5d.ckpt')
    for i, (datas, labells) in enumerate(validate_generator):
        feed_dict[inputs] = datas
        res = sess.run(feature,feed_dict=feed_dict)
        all_activations['final'].append(res)
        if i >= num_validate - 1:
            break

for key in all_activations:
    all_activations[key] = np.concatenate(all_activations[key])
    print(all_activations[key].shape)
with open('/home/pr606/python_vir/yuan/EXTRA_DATA/r2.5d_512_features.pickle', 'wb') as handle:
        pickle.dump(all_activations, handle)


