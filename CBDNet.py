import os, time, scipy.io, shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # 3 ==> 32 的输入卷积
        self.inc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True))
        
        # 32 ==> 32 的中间卷积
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 32 ==> 3 的输出卷积 
        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 第 1 次卷积
        conv1 = self.inc(x)
        # 第 2 次卷积
        conv2 = self.conv(conv1)
        # 第 3 次卷积
        conv3 = self.conv(conv2)
        # 第 4 次卷积
        conv4 = self.conv(conv3)
        # 第 5 次卷积
        conv5 = self.outc(conv4)
        return conv5

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    # forward 需要两个输入，x1 是需要上采样的小尺寸 feature map
    # x2 是以前的大尺寸 feature map，因为中间的 pooling 可能损失了边缘像素，
    # 所以上采样以后的 x1 可能会比 x2 尺寸小
    def forward(self, x1, x2):
        # x1 上采样
        x1 = self.up(x1)
        
        # 输入数据是四维的，第一个维度是样本数，剩下的三个维度是 CHW
        # 所以 Y 方向上的悄寸差别在 [2],  X 方向上的尺寸差别在 [3] 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 给 x1 进行 padding 操作
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        # 把 x2 加到反卷积后的 feature map
        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64))

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128))

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256))

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128))

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64))

        self.outc = outconv(64, 3)

    def forward(self, x):
        # input conv : 6 ==> 64 ==> 64
        inx = self.inc(x)

        # 均值 pooling, 然后 conv1 : 64 ==> 128 ==> 128 ==> 128
        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        # 均值 pooling，然后 conv2 : 128 ==> 256 ==> 256 ==> 256 ==> 256 ==> 256 ==> 256
        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        # up1 : conv2 反卷积，和 conv1 的结果相加，输入256，输出128
        up1 = self.up1(conv2, conv1)
        # conv3 : 128 ==> 128 ==> 128 ==> 128
        conv3 = self.conv3(up1)

        # up2 : conv3 反卷积，和 input conv 的结果相加，输入128，输出64
        up2 = self.up2(conv3, inx)
        # conv4 : 64 ==> 64 ==> 64
        conv4 = self.conv4(up2)

        # output conv: 65 ==> 3，用1x1的卷积降维，得到降噪结果
        out = self.outc(conv4)
        return out

class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()
    
    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out