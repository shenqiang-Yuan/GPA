import os
# from scipy import misc
import numpy as np
# from PIL import Image
import json
import functools
import collections
# from skimage import transform
# from skimage import io
import re
import random
import cv2


def pil_loader(path, scale, area=None):
    # scale=(240,320),size=(224,224)
    # scale=(168,224),size=(112,112)
    # cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2RGB
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(img, dsize=(scale[1],scale[0]),interpolation=cv2.INTER_CUBIC)
    if area:
        top, left, bottom, right = area
        frame = frame[top:bottom, left:right]
    else:
        pass
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img
    """
    return frame


def io_loader(path):
    return io.imread(path)


def get_default_image_loader():
    return pil_loader


def video_loader(video_dir_path, Suffix, frame_indices, image_loader,scale,size=(112,112), crop=None):
    """
    :param video_dir_path:
    :param Suffix:
    :param frame_indices:
    :param image_loader:
    :param scale:
    :param size:
    :param crop: if not None,size must be specified,and satisfied scale[0]-size[0]>0,scale[1]-size[1]>0,
    :return:
    """
    video = []
    if crop == 'center':
        top = (scale[0]-size[0])//2
        left = (scale[1]-size[1])//2
        bottom = top+size[0]
        right = left+size[1]
        area = (top, left, bottom, right)
    elif crop == 'random':
        top = random.randint(0,(scale[0]-size[0])//2)
        left = random.randint(0,(scale[1]-size[1])//2)
        bottom = top+size[0]
        right = left+size[1]
        area = (top,left,bottom,right)
    else:
        area=None
    for i in frame_indices:
        '''if satrt_frame is 0, error will occurs,so set offset=1 to avoid this situation'''
        image_path = os.path.join(video_dir_path, 'image_%05d.%s' % (i+1, Suffix))
        if os.path.exists(image_path):
            video.append(image_loader(image_path,scale,area))
        else:
            print("warning ,this clip is empty {}, with frame_indices {}".format(video_dir_path,frame_indices))
            return video

    return video


def get_default_video_loader(crop,scale,size):
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader,crop=crop,scale=scale,size=size)


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(112),
        >>>     transforms.CenterCrop(100),
        >>>     transforms.Normalize([114.7748, 107.7354, 99.4750],[1,1,1],(100,100,3))
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Scale(object):
    """Rescale the input PIL.Image or np.ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, mode='constant'):
        '''
        :param size: (h, w)
        :param mode: 'constant', 'reflect', 'nearest','wrap' and 'mirror',
        'bicubic' or 'cubic'
        '''
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        self.size = size
        self.mode = mode

    def __call__(self, img):
        """
        Args:
            img PIL.Image: Image to be scaled. WxH,but
            changed to be skimage.io loader,so img is an ndarray object with shape=(h,w,c)
        Returns:
           np.ndarray: Rescaled image.
        """
        if isinstance(self.size, int):
            if isinstance(img, Image.Image):
                w, h = img.size
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return img
                if w < h:
                    ow = self.size
                    oh = int(self.size * float(h) / w)
                    img = img.resize((ow, oh), Image.NEAREST)
                    return np.asarray(img, dtype=np.float32)
                else:
                    oh = self.size
                    ow = int(self.size * float(w) / h)
                    img = img.resize((ow, oh), Image.NEAREST)
                    return np.asarray(img, dtype=np.float32)
            elif isinstance(img, np.ndarray):
                h, w, c = img.shape
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return img
                if w < h:
                    ow = self.size
                    oh = int(self.size * float(h) / w)
                    return transform.resize(img, (oh, ow), mode=self.mode, anti_aliasing=False)
                    # scale = self.size / float(w)
                    # return transform.rescale(img, scale)
                else:
                    oh = self.size
                    ow = int(self.size * float(w) / h)
                    return transform.resize(img, (oh, ow), mode=self.mode, anti_aliasing=False)
                    # scale = self.size / float(h)
                    # return transform.rescale(img, scale)
            else:
                raise ValueError('img must be PIL.Image or np.ndarray')
        else:
            # PIL加载的图片，size第一个数是宽w，第二个才是高w，
            # 而misc/io/matplot默认第一个数是高h，第二个是宽w
            if isinstance(img, Image.Image):
                img = img.resize((self.size[1], self.size[0]), Image.NEAREST)
                return np.asarray(img, dtype=np.float32)
            elif isinstance(img, np.ndarray):
                return transform.resize(img, (self.size[0], self.size[1]), mode=self.mode, anti_aliasing=False)
                # h_scale = self.size[0] / float(img.shape[0])
                # w_scale = self.size[1] / float(img.shape[1])
                # return transform.rescale(img, (h_scale, w_scale))
            else:
                raise ValueError('img must be PIL.Image or np.ndarray')


class CenterCrop(object):
    """Crops the given ndarray image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img nd.array: Image to be cropped.WxHxC
        Returns:
            np.ndarray: Cropped image.
        """
        w, h, _ = img.shape
        th, tw = self.size
        startx = int(round(w-tw)/2.)      # round() 四舍五入
        starty = int(round(h-th)/2.)
        return img[starty:starty + th, startx:startx + tw]


class Normalize(object):
    """Normalize an ndarray image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std, img_size, img_channel):
        assert len(img_size) == 2 and (img_channel == 1 or img_channel == 3)
        self.channel = img_channel
        if self.channel == 3 and len(mean) == 3 and len(std) == 3:
            mean1 = mean[0] * np.ones(shape=(img_size[0], img_size[1], 1), dtype=np.float32)
            mean2 = mean[1] * np.ones(shape=(img_size[0], img_size[1], 1), dtype=np.float32)
            mean3 = mean[2] * np.ones(shape=(img_size[0], img_size[1], 1), dtype=np.float32)
            std1 = std[0] * np.ones(shape=(img_size[0], img_size[1], 1), dtype=np.float32)
            std2 = std[0] * np.ones(shape=(img_size[0], img_size[1], 1), dtype=np.float32)
            std3 = std[0] * np.ones(shape=(img_size[0], img_size[1], 1), dtype=np.float32)
            self.mean = np.concatenate((mean1, mean2, mean3), axis=2)
            self.std = np.concatenate((std1, std2, std3), axis=2)

        elif self.channel == 1 and len(mean) == 1 and len(std) == 1:
            self.mean = mean[0] * np.ones(shape=(img_size[0], img_size[1]), dtype=np.float32)
            self.std = std[0] * np.ones(shape=(img_size[0], img_size[1]), dtype=np.float32)
        else:
            raise ValueError('image shape must be WxHxC or WxHxC, C must be 1 or 3')

    def __call__(self, image):
        """
        Args:
            tensor (Tensor): Tensor image of size (H, W, C) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        assert isinstance(image, np.ndarray)
        image = np.asarray(image,dtype=np.float32)
        image = image - self.mean
        image = image / self.std
        return image


class DataSet(object):
    def __init__(self, clip_length=16, sample_step=1,
                 data_root='', annotation_path='', spatial_transform=None, mode='train',with_start=None,multi_sample=False):
        assert os.path.exists(annotation_path), "path is not exists"
        with open(annotation_path) as f1:
            annotation_json = json.load(f1)
        data_json = annotation_json["database"]
        label_list = annotation_json["labels"]
        data_description = {}
        
        if mode == 'train':
            for (key, value) in data_json.items():
                if value['subset'] == "training":
                    data_description.update({key: value})
        elif mode == 'validation':
            for (key, value) in data_json.items():
                if value['subset'] == "validation":
                    data_description.update({key: value})
        elif mode == 'test':
            for (key, value) in data_json.items():
                if value['subset'] == "test":
                    data_description.update({key: value})
        elif mode == None:
            for (key, value) in data_json.items():
                data_description.update({key: value})
        else:
            raise ValueError("mode must be train/validation/test")
        
        self.classes = label_list
        self.num_class = len(label_list)
        self.data = data_description
        self.data_root = data_root
        self.loader = get_default_video_loader(crop='center',scale=(168,224),size=(112,112))
        self.length = clip_length
        self.sample_step = int(sample_step)
        self.spatial_transform = spatial_transform
        self.samples_name = list(data_description.keys())
        self.amount = len(data_description.keys())
        self.start_frame = with_start
        self.multi_sample = multi_sample
        self.name_pattern = re.compile('\_\d*$')
        if multi_sample:
            assert self.start_frame,'multi sampling per video, must set a start frame index'

    def __getitem__(self, key):
        if isinstance(key, str):
            name = key
            _sample = self.data[name]
        else:
            name = self.samples_name[key]
            _sample = self.data[name]
        sample_label = _sample["annotations"]['label']
        sample_depth = _sample["annotations"]["n_frames"]

        sample_start = None if not self.start_frame else _sample["annotations"]["start_fr"]
        name = name if not self.start_frame else self.name_pattern.sub("",name)
        sample_path = os.path.join(self.data_root, sample_label, name)
        if not sample_start:
            if self.sample_step*self.length < sample_depth:
                begins = np.random.randint(0, sample_depth - self.sample_step*self.length + 1)
                image_indexs = [i for i in range(begins, self.sample_step*self.length + begins, self.sample_step)]
            elif self.length < sample_depth:
                begins = np.random.randint(0, sample_depth - self.length + 1)
                image_indexs = [i for i in range(begins, self.length + begins)]
            else:
                image_indexs = [i for i in range(sample_depth)]
        else:
            image_indexs = [i for i in range(sample_start,
                            min(self.length*self.sample_step + sample_start,
                            sample_depth),
                            self.sample_step)]

        for index in image_indexs:
            if len(image_indexs) >= self.length:
                break
            image_indexs.append(index)
        # image_names = os.listdir(sample_path)

        try:
            sample = self.loader(sample_path, 'jpg', image_indexs)
            '''
            if self.spatial_transform is not None:
                sample = [self.spatial_transform(img) for img in sample]
            else:
                sample = [np.asarray(img, dtype=np.float32) for img in sample]
            '''
            sample = np.asarray(sample)
            """convert class_name to number, from zero to num_class-1 """
            label = self.classes.index(sample_label)
            label_one_hot = np.zeros(shape=(self.num_class,), dtype='int')
            label_one_hot[int(label)] = 1
            # print('this clip is from ',sample_path)
            return sample, label_one_hot
        except Exception as e:
            print("read sample from disk error")
            raise e


def batch_data_loader(dataset, batchsize=1, triplast=False):
    assert isinstance(dataset, DataSet)
    epoch_ending = False
    shape = (batchsize,) + dataset.sample_size
    while not epoch_ending:
        batch = np.zeros(shape=shape)
        labels = np.zeros(shape=(batchsize, 1))
        for i in range(batchsize):
            data = next(dataset)
            if isinstance(data, tuple):
                batch[i] = data[0]
                labels[i] = data[1]
            else:
                epoch_ending = True
                batch = batch[0:i]
                labels = labels[0:i]
                break
        if epoch_ending and triplast:
            break
        yield batch, labels


if __name__ == '__main__':

    """
        warning:
            这个文件内的代码可以在不同深度深度学习框架下通用，用以批量加载图片数据集
            或者视频数据集（视频数据已切分成图片），但是没有实现多线程模式，因此加载
            速度受到限制。若追求更快的训练速度，推荐使用pytorch自带的DataLoader类代替
           batch_data_loader() 实现快速的数据加载,把pytorch当做通用的numpy使用
    """
    spatial_transform = Compose([Scale((224, 224))])
    json_file1 = "/home/pr606/python_vir/yuan/i3d-kinects/dataset/ucf101_lmdb.json"
    json_file2 = '/home/pr606/Pictures/dataset_annotations/ucf101_json_file/ucf101_01.json'

    train_set = DataSet(clip_length=25,
                        sample_step=9,
                        data_root='/home/pr606/Pictures/UCF101DATASET/ucf101',
                        annotation_path=json_file1,
                        spatial_transform=spatial_transform,
                        mode='train',
                        with_start=True, multi_sample=True)
