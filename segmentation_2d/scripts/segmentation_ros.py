#!/usr/bin/env python
# for ros
import rospy
from std_msgs.msg import String 
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from copy import deepcopy

# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.io import loadmat

# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
import sys

DIR_PATH = os.path.dirname(sys.path[0])

colors = loadmat(DIR_PATH+'/scripts/data/color150.mat')['colors']
# road 7 
colors[6] = [128,  64, 128]
# sidewalk 12
colors[11] = [244,  35, 232]
# Building 2
colors[1] = [70,  70,  70]
# wall 1
colors[0] = [102, 102, 156]
# Fence 33
colors[32] = [190, 153, 153]
# Pole 94
colors[93] = [153, 153, 153]
# traffic 137
colors[136] = [250, 170,  30]
# sky 3
colors[2] = [70, 130, 180]
# person 13
colors[12] = [220,  20,  60]
# cycle 128
colors[127] = [255,   0,   0]
# car 21
colors[20] = [0,   0, 142]
# Truck 84
colors[83] = [ 0,   0,  70]
# Bus 81
colors[80] = [0,  60, 100]

def load_model(args):
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights_encoder = os.path.join(DIR_PATH, 'models', args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(DIR_PATH, 'models', args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    torch.cuda.set_device(args.gpu)
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    if torch.cuda.is_available():
        segmentation_module.cuda()
    return segmentation_module

class Segmentation(object):
    def __init__(self, segmentation_module, nums_class, padding_constant, rate=2):
        self.image_sub = rospy.Subscriber("/kitti/camera_color_left/image_raw", Image, self.imageCallback, queue_size=2)
        self.image_pub = rospy.Publisher("/semantic_segmentation/image", Image, queue_size=2)
        self.segmentation_module = segmentation_module
        self.padding_constant = padding_constant
        self.nums_class = nums_class
        self.normalize = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.])
        self.flag = False
        rospy.init_node('segmentation_node')
        rate = rospy.Rate(rate)
        
        while not rospy.is_shutdown():
            if self.flag:
                self.segmentation_frame(self.cv_image)
                self.flag = False

    def imageCallback(self, image_msg):
        self.cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
        self.frame_id = image_msg.header.frame_id
	self.stamp = image_msg.header.stamp
        self.flag = True

        #self.segmentation_frame(cv_image)
        
    def segmentation_frame(self, img):
        #image = cv2.resize(image, (height, width))
        ori_height, ori_width, _ = img.shape

        this_short_size = 600
        imgMaxSize = 1000
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_height = self.round2nearest_multiple(target_height, self.padding_constant)
        target_width = self.round2nearest_multiple(target_width, self.padding_constant)

        # resize
        image = cv2.resize(img.copy(), (target_width, target_height))

        # image transform
        image = self.img_transform(image).cuda()

        image = torch.unsqueeze(image, 0)

        #image = torch.tensor(image).

        print ('image ', image.shape)

        #segSize = (image.shape[2],image.shape[3])
        segSize = (ori_height, ori_width)
        scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        scores = async_copy_to(scores, args.gpu)
        feed_dict = {}
        feed_dict['img_data'] = image
        # forward pass
        pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
        scores = scores + pred_tmp
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())
        print ('pred ', pred.shape)
        pred_color = colorEncode(pred, colors).astype(np.uint8)
        pub_image = CvBridge().cv2_to_imgmsg(pred_color, "bgr8")
	pub_image.header.frame_id = self.frame_id
        pub_image.header.stamp = self.stamp
        self.image_pub.publish(pub_image)
    
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def img_transform(self, img):
        # image to float
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img


def main(args):
    segmentation_module = load_model(args)
    segmentation = Segmentation(segmentation_module, args.num_class, args.padding_constant)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    # parser.add_argument('--test_imgs', required=True, nargs='+', type=str,
    #                     help='a list of image paths, or a directory name')
#baseline-resnet50dilated-ppm_deepsup
#baseline-mobilenetv2dilated-c1_deepsup
    parser.add_argument('--model_path', default='baseline-resnet50dilated-ppm_deepsup',
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')

    args = parser.parse_args()

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
