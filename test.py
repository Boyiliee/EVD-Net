import os
import numpy as np
import caffe
import sys
import re
import random
import time
import copy
import cv2
import scipy
import shutil
import csv
from PIL import Image
import datetime


def EditFcnProto(templateFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
        print templateFile
        outFile = 'DeployT.prototxt'
        with open(outFile, 'w') as fd:
            fd.write(template.format(height=height,width=width))

def test():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #caffe.set_mode_cpu();

    info = os.listdir(r'VIDEO_test_img');

    model = r'EVD-Net.caffemodel'

    net = caffe.Net('test.prototxt', model, caffe.TEST);

    imagesnum=0;
    for line in info:
        reg = re.compile(r'(.*?).jpg');
        all = reg.findall(line)
        if (all != []):
            imagename = str(all[0]);
            line=imagename
            reg = re.compile(r'ILSVRC2015_train_00124006_([0-9]{6})_1_3');
            all = reg.findall(line)
            labelnum = int(all[0]);
            if (os.path.isfile(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum-1).zfill(6)) == False or
                os.path.isfile(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum-2).zfill(6)) == False or
                os.path.isfile(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum+1).zfill(6)) == False or
                os.path.isfile(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum+2).zfill(6)) == False):
                continue;
            else:
                imagesnum = imagesnum + 1;

                npstore_1 = caffe.io.load_image(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum-2).zfill(6))
                npstore_2 = caffe.io.load_image(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum-1).zfill(6))
                npstore = caffe.io.load_image(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum).zfill(6))
                npstore_3 = caffe.io.load_image(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum+1).zfill(6))
                npstore_4 = caffe.io.load_image(r'VIDEO_test_img\ILSVRC2015_train_00124006_%s_1_3.jpg' % str(labelnum+2).zfill(6))



                batchdata = []
                data = npstore_1
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['img_1'].data[...] = batchdata;
                batchdata = []
                data = npstore_2
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['img_2'].data[...] = batchdata;
                batchdata = []
                data = npstore
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['img'].data[...] = batchdata;
                batchdata = []
                data = npstore_3
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['img_3'].data[...] = batchdata;
                batchdata = []
                data = npstore_4
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['img_4'].data[...] = batchdata;
                net.forward()
                data = net.blobs['sum'].data[0];
                data = data.transpose((1, 2, 0));
                data = data[:, :, ::-1]


                savepath = 'result\\' + imagename + '_EVD-Net.jpg'
                cv2.imwrite(savepath, data * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])

                print imagename

        print 'image numbers:',imagesnum;

def main():
    test()


if __name__ == '__main__':
    main();


