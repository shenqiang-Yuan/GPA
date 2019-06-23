import numpy as np
import matplotlib.pyplot as plt
from itertools import product as product
import cv2
import json,pickle
from samples import read_clips


def restore_resolution(feat,reresolution=None,offset=0.5,dtype=np.float32,prefill=0):
    if reresolution:
        (W,H,C) = feat.shape
        ret = np.empty(reresolution +(C,), dtype=dtype)
        ret.fill(prefill)
        scale_h = float(W)/reresolution[0]
        scale_w = float(H)/reresolution[1]
        # rows, columns = product(range(reresolution[0]), repeat=2)
        rows = range(reresolution[0])
        columns = range(reresolution[1])
        for k in range(C):
            for row in rows:
                for column in columns:
                    src_norm_h = (row + offset) * scale_h - offset
                    src_norm_w = (column + offset) * scale_w - offset
                    # print(src_norm_h,src_norm_w)
                    src_h_0 = int(np.floor(src_norm_h))
                    src_w_0 = int(np.floor(src_norm_w))
                    src_h_1 = min(src_h_0 + 1, H - 1)
                    src_w_1 = min(src_w_0 + 1, W - 1)

                    interplation_w0 = (src_w_1 - src_norm_w) * feat[src_h_0, src_w_0, k] + (src_norm_w - src_w_0) * feat[src_h_0, src_w_1, k]
                    interplation_w1 = (src_w_1 - src_norm_w) * feat[src_h_1, src_w_0, k] + (src_norm_w - src_w_0) * feat[src_h_1, src_w_1, k]
                
                    ret[row, column, k] = (src_h_1 - src_norm_h) * interplation_w0 + (src_norm_h - src_h_0) * interplation_w1
                
        return ret.astype(np.int8)
    else:
        return feat.astype(np.int8)


def scale_center_crop(src,scale,size):
    src = cv2.resize(src, dsize=(scale[1], scale[0]), interpolation=cv2.INTER_CUBIC)
    top = (scale[0] - size[0]) // 2
    left = (scale[1] - size[1]) // 2
    bottom = top + size[0]
    right = left + size[1]

    frame = src[top:bottom, left:right]

    return frame


if __name__=='__main__':
    import matplotlib.pyplot as plt
    video_path = '/home/pr606/Pictures/ucf_images/GolfSwing/v_GolfSwing_g04_c03/'
    path = r"/home/pr606/YUAN/history/tf-R2plus1D/samples/v_GolfSwing_g04_c03.pickle"
    with open(path,'rb') as f1:
        features = pickle.load(f1)
    feat2 = features['com2_gpa']
    feat4 = features['com4_gpa']

    oneclip = feat4[1]
    oneclip = oneclip.sum(axis=3)
    # img = cv2.cvtColor(feat, cv2.COLOR_BGR2GRAY)
    for ind in range(oneclip.shape[0]):
        slice = cv2.resize(oneclip[ind], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        slice = cv2.Laplacian(slice, ddepth=-1, ksize=3)
        img = cv2.imread(video_path+'image_%05d.jpg' % (31+2*ind+1))
        img = scale_center_crop(img, scale=(240, 280), size=(224, 224))
        img[:,:,2] = img[:,:,2] + slice
        # img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
        # slice = cv2.convertScaleAbs(slice)
        # plt.imshow(slice)
        # plt.show()
        img *= (img > 0.0)
        img = img *(img < 255.0) + 255.0*(img > 255.0)
        img = img.astype(np.uint8)
        plt.axis('off')
        plt.imsave('v_GolfSwing_g04_c03/gpa4-32-63/applied/image_%05d.png' % (31+2*ind+1),img)
        # plt.show()
        # cv2.imshow('result', img)
        # cv2.waitKey(500)
        # cv2.imwrite('v_Skijet_g03_c02/image_%05d.jpg' % (31+2*ind+1),img)
    pass
