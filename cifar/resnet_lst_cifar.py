'''
LST-Net under ResNet & WRN for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
[3] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.
[4] L. Li, K. Wang, S. Li, X. Feng and L. Zhang. LST-Net: Learning a Convolutional Neural Network with a Learnable Sparse Transform. In ECCV, 2020.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .dct import dct_init

class ST(nn.Module):
    def __init__(self, tau=1e-4):
        super(ST,self).__init__()
        self.tau = tau

    def forward(self, x):
        y = torch.where(torch.abs(x) < self.tau,
                        torch.zeros_like(x),
                        torch.where(x < 0,
                                    x + self.tau,
                                    x - self.tau))
        return y

class HT(nn.Module):
    def __init__(self, tau=1e-4):
        super(HT,self).__init__()
        self.tau = tau

    def forward(self, x):
        y = torch.where(torch.abs(x) < self.tau,
                        torch.zeros_like(x),
                        x)
        return y

class LST2Block(nn.Module):
    expansion = 1

    def __init__(self, in_chnls, out_chnls, stride=1, downsample=None, k=3, a=2, tau=1e-4):
        self.stride = stride
        self.in_chnls = in_chnls
        self.out_chnls = out_chnls
        self.k = k
        self.a = max(2, min(a, k))
        self.padding = (self.k-1) // 2 #padding='same'
        self.tau = tau

        super(LST2Block, self).__init__()
        self.downsample = downsample

        self.base_chnls = self.out_chnls // self.a // self.a

        #self.weight1 = nn.Parameter(torch.FloatTensor(self.base_chnls, self.in_chnls, 1, 1))
        self.conv1 = nn.Conv2d(self.in_chnls, self.base_chnls, 1, bias=False)
        self.weight2 = nn.Parameter(torch.FloatTensor(self.base_chnls, self.base_chnls, 1, 1))
        self.weight3 = nn.Parameter(torch.FloatTensor(self.a*self.a,1,self.k,self.k))

        self.bn1 = nn.BatchNorm2d(self.base_chnls)
        self.bn2 = nn.BatchNorm2d(self.base_chnls)
        self.bn3 = nn.BatchNorm2d(self.out_chnls)
        self.activation = nn.ReLU(inplace=True)

        self.init_param()

    def init_param(self):
        # weight2
        self.weight2.data = dct_init(self.base_chnls, self.base_chnls).view(self.weight2.shape)

        # weight3
        part_a = dct_init(self.a, self.k).view(self.a, 1, 1, self.k)
        part_b = torch.transpose(part_a.clone(), -1, -2)
        part_a = part_a.repeat(1,1,self.k,1)
        part_b = part_b.repeat(1,1,1,self.k)

        idx = 0
        for i in range(self.a):
            for j in range(self.a):
                self.weight3.data[idx,:,:,:] = part_a[i,:,:,:] * part_b[j,:,:,:]
                idx += 1

        # bn1
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()

        # bn2
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

        # bn3
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()

    def forward(self, x):
        residual = x
        y1 = residual

        if self.downsample is not None:
            y1 = self.downsample(residual)

        y2 = self.conv1(residual)
        y2 = self.bn1(y2)
        y2 = self.activation(y2)

        y2 = F.conv2d(y2, self.weight2)
        y2 = self.bn2(y2)
        y2 = torch.where(torch.abs(y2) < self.tau,
                         torch.zeros_like(y2),
                         torch.where(y2 < 0,
                                     y2 + self.tau,
                                     y2 - self.tau))


        w = self.weight3.repeat(self.base_chnls, 1, 1, 1)
        y2 = F.conv2d(y2, 
                      w, 
                      stride=self.stride, 
                      padding=self.padding, 
                      groups=self.base_chnls)
        y2 = self.bn3(y2)

        y = y1 + y2
        y = torch.where(torch.abs(y) < self.tau,
                        torch.zeros_like(y),
                        torch.where(y < 0,
                                    y + self.tau,
                                    y - self.tau))
        return y
    

class ResNet_LST_Cifar(nn.Module):
    init_width = 32
    init_height= 32

    def __init__(self, block, layers, num_classes=10, k=3, a=2, tau=1e-4):
        super(ResNet_LST_Cifar, self).__init__()
        self.k = k
        self.a = a
        self.tau = tau
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])

        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.handler = nn.Sequential(nn.Conv2d(64*a*a, 64, 1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.a * self.a:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*self.a*self.a,1,stride=stride,bias=False, groups=self.inplanes),
                                       nn.BatchNorm2d(planes*self.a*self.a))

        layers = []
        layers.append(block(self.inplanes,
                            planes*self.a*self.a,
                            stride,
                            downsample,
                            tau=self.tau,
                            k=self.k,
                            a=self.a))

        self.inplanes = planes * self.a * self.a
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, tau=self.tau, k=self.k, a=self.a))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.handler(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class WRN_LST_Cifar(nn.Module):
    init_width = 32
    init_height= 32

    def __init__(self, block, layers, num_classes=10, k=3, a=2, width=10, tau=1e-4):
        super(WRN_LST_Cifar, self).__init__()

        self.width = width
        self.k = k
        self.a = a
        self.num_classes = num_classes
        self.tau = tau
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16*width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32*width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*width, layers[2], stride=2)

        self.handler = nn.Sequential(nn.Conv2d(64*width*a*a, 64*width, 1, bias=False),
                                     nn.BatchNorm2d(64*width),
                                     nn.ReLU(inplace=True))

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.a * self.a:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*self.a*self.a,1,stride=stride,bias=False, groups=self.inplanes),
                                       nn.BatchNorm2d(planes*self.a*self.a))

        layers = []
        layers.append(block(self.inplanes,
                            planes*self.a*self.a,
                            stride,
                            downsample,
                            tau=self.tau,
                            k=self.k,
                            a=self.a))

        self.inplanes = planes * self.a * self.a
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, k=self.k, a=self.a, tau=self.tau))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        print('before handler', x.shape)
        x = self.handler(x)
        print('after handler', x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print('after gap', x.shape)
        x = self.fc(x)

        return x

def wrn16_8_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [2,2,2]], width=8, k=k, a=a, **kwargs)


def wrn16_10_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [2,2,2]], k=k, a=a, **kwargs)

def wrn22_8_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [3,3,3]], width=8, k=k, a=a, **kwargs)

def wrn22_10_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [3,3,3]], k=k, a=a, **kwargs)

def wrn28_10_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [4,4,4]], k=k, a=a, **kwargs)

def wrn28_12_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [4,4,4]], width=12, k=k, a=a, **kwargs)

def wrn40_4_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [6,6,6]], width=4, k=k, a=a, **kwargs)

def wrn40_8_lst_cifar(k=3,a=2, **kwargs):
    replica = 2
    return WRN_LST_Cifar(LST2Block, [x*replica for x in [6,6,6]], width=8, k=k, a=a, **kwargs)

def resnet20_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [3,3,3]], k=k, a=a, **kwargs)
    return model

def resnet32_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [5,5,5]], k=k, a=a, **kwargs)
    return model

def resnet44_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [7,7,7]], k=k, a=a, **kwargs)
    return model

def resnet56_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [9,9,9]], k=k, a=a, **kwargs)
    return model

def resnet110_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [18,18,18]], k=k, a=a, **kwargs)
    return model

def resnet164_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [27,27,27]], k=k, a=a, **kwargs)
    return model

def resnet1202_lst_cifar(k=3, a=2, **kwargs):
    replica = 2
    model = ResNet_LST_Cifar(LST2Block, [x*replica for x in [200,200,200]], k=k, a=a, **kwargs)
    return model