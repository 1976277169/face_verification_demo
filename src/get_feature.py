# -*- coding: utf-8 -*-
__author__ = 'lufo'

# this code can get image's feature using CNN

import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import common
from scipy.io import savemat

sys.path.append('/home/liuxuebo/anaconda2/lib/python2.7/site-packages')

# Make sure that caffe is on the python path:
caffe_root = '/home/liuxuebo/CV/caffe-master/'  # this file is expected to be in {caffe_root}/examples

sys.path.insert(0, caffe_root + 'python')

import caffe


def caffe_fea_extr(net, img_list, fea_dim, img_h, img_w, batch_num):
    """
    extract and save images' feature using caffe
    :param net: caffe net
    :param img_list: every elem is an image's path
    :param fea_dim: dimension of feature extract by caffe
    :param batch_num: number of images in a batch
    :param img_h: height of origin image
    :param img_w: width of origin image
    :param result_fold: save features in this path
    :return: feature list
    """
    fea_all = np.zeros((len(img_list) + batch_num, fea_dim))
    batch_list = (img_list[i:i + batch_num] for i in range(0, len(img_list), batch_num))
    img_batch = np.zeros((batch_num, 1, img_h, img_w))
    for j, batch in enumerate(batch_list):
        for i, img_path in enumerate(batch):
            img = common.trans_img_to_grey(Image.open(img_path))
            img_batch[i, 0, :, :] = img.resize((img_h, img_w), Image.BILINEAR)
        net.blobs['data'].data[...] = img_batch
        fea = net.forward()['pool5/7x7_s1']
        fea_all[(j * batch_num):((j + 1) * batch_num), :] = fea[:, :, 0, 0]
        # for item in ((k, v.data.shape) for k, v in net.blobs.items()):
        #    print(item)
    fea_all = fea_all[:len(img_list)]
    return fea_all


def get_feature(img_list):
    # get caffe net
    caffe.set_device(0)
    caffe.set_mode_gpu()  # set gpu model
    net = caffe.Net('/home/liuxuebo/CV/BLUFR/deploy.txt',
                    '/home/liuxuebo/CV/BLUFR/models/nosym_iter_500000.caffemodel',
                    caffe.TEST)

    return caffe_fea_extr(net, img_list, 896, 128, 128, 128)


def save_lfw_feature(img_list, result_fold='../data/'):
    fea_all = get_feature(img_list)
    savemat(result_fold + 'googlenet_lfw.mat', {'googlenet_lfw': fea_all})


if __name__ == '__main__':
    # get image list
    img_list = []
    with open('/home/liuxuebo/CV/BLUFR/lfw_nosym_list.txt') as fr:
        for image_path in fr.readlines():
            img_list.append(image_path.strip())
    save_lfw_feature()
