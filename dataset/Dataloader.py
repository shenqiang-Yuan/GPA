from tensorflow.python.keras.utils import Sequence
import numpy as np
import math
import importlib
import pandas as pd
import sys
import time
sys.path.append("..")
from dataset import DataSet


class DataGenerator(Sequence):
    # Sequence会管理这个迭代器，一旦迭代完成(index>len-1)，又会开
    # 启新一轮迭代,所以在训练时要设定每轮的迭代次数，
    # 即多少个batch结束一轮
    # 至少需要实现__len__()和__getitem__()两个函数
    # 可仿照pytorch写一个
    def __init__(self, Data,  batch_size=1, shuffle=True, ordered_file_path=None):
        assert isinstance(Data, DataSet)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = Data
        self.multi_sample = Data.multi_sample
        self.video_names_lmdb = None
        self.iterations = self.__len__()
        if self.multi_sample:
            assert ordered_file_path,'must have a ordered video names file'
            self.video_names_lmdb = pd.read_csv(ordered_file_path,
                                                          delimiter=None,
                                                          header=None)[0]
    def __len__(self):
        # 计算batches_per_epoch
        # math.ceil()向上取整数
        return int(self.dataset.amount / float(self.batch_size))

    def __getitem__(self, index):
        if index>self.iterations:
            raise StopIteration
        if not self.multi_sample:
            # index表示获取第index个batch数据
            # 生成batch_size个样本索引号
            if self.shuffle:
                indexs = np.random.rand(self.batch_size) * (self.dataset.amount+50)
                indexs = np.clip(indexs, 0., self.dataset.amount - 1)
            else:
                if (index + 1) * self.batch_size <= self.dataset.amount:
                    indexs = range(index * self.batch_size, (index + 1) * self.batch_size, 1)
                else:
                    indexs = np.random.randint(index*self.batch_size, self.dataset.amount, size=self.batch_size)
            # 根据索引获取dataset集合中的数据
            Batches = [self.dataset[math.ceil(k)] for k in indexs]
            # videos_name = [self.dataset.samples_name[math.ceil(k)] for k in indexs]
        else:
            # 根据视频的名字获取一个batch的数据
            # if (index + 1) * self.batch_size > self.dataset.amount:
            #     print("data running out,please satrt a new epoch")
            #     index=0
            videos_name = self.video_names_lmdb[index*self.batch_size:(index+1)*self.batch_size]
            Batches = [self.dataset[key] for key in videos_name]
        # 生成batch数据
        # datas = []
        # targets = []
        datas, targets = zip(*Batches)
        # for i, (data, target) in enumerate(Batches):
        #     # data must be a ndarray
        #     # target must be a ndarray,and encode as one-hot (1,num_class)
        #     datas.append(data)
        #     targets.append(target)
        return np.asarray(datas), np.asarray(targets)#,np.asarray(videos_name)


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    from dataset import Dataset, Compose, Scale, CenterCrop, Normalize
    import scipy.misc as misc
    # mean=[114.7748, 107.7354, 99.4750]
    # std=[58.395, 57.12, 57.375]

    spatial_transform = Compose([Scale((224, 224))])
    json_file1 = "/home/pr606/python_vir/yuan/i3d-kinects/dataset/ucf101_lmdb.json"
    json_file2 = '/home/pr606/Pictures/dataset_annotations/ucf101_json_file/ucf101_01.json'

    train_set = DataSet(clip_length=25,
                        sample_step=9,
                        data_root='/home/pr606/Pictures/UCF101DATASET/ucf101',
                        annotation_path=json_file1,
                        spatial_transform=None,
                        mode='train',
                        with_start=True, multi_sample=True)

    train_generator = DataGenerator(train_set, batch_size=7, ordered_file_path='./names_in_order.csv')
    print(train_generator.__len__())   # NUM_of_datas / batchsize
    for i,(datas,labels) in enumerate(train_generator):
        # print(datas.shape)
        # length = datas[0].shape[0]
        # print(datas.shape,labels.shape)
        # for i in range(length):
        #     _img = datas[3][i]
        #     img = datas[3][i]
        #     #'''还原被归一化的图片'''
        #     # img[:,:,0] = img[:,:,0]*58.395 + 114.7748
        #     # img[:,:,1] = img[:,:,1]*57.12 + 107.7354
        #     # img[:,:,2] = img[:,:,2]*57.375 + 99.4750
        #     img = Image.fromarray(np.uint8(_img))
        #     img.show()
        # break
        # time.sleep(1)
        print(datas.shape,datas[0, 0, 0:5, 0:5, 0:3])
        img = Image.fromarray(np.uint8(datas[0, 0, :, :, :]))
        img.show()
        pass
    print('ending!')
