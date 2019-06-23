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

    return frame


def get_default_image_loader():
    return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader,scale, Suffix,size=(112,112), crop='center'):
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


def get_batch(pathes,index_list):
    assert len(pathes) == len(index_list)
    frame_loader = get_default_image_loader()
    batchs = []
    for vi in range(len(pathes)):
        batchs.append(video_loader(pathes[vi],index_list[vi],frame_loader,
                                   scale=(240, 280),Suffix='jpg',size=(224,224)))
    return np.asarray(batchs)