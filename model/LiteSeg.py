import math

import torch
from caffe2.python.models import shufflenet
from torch import nn
import torch.nn.functional as F

from torchvision.models import mobilenet

from model.base_model import darknet



def build(backbone_network ,modelpath ,CONFIG ,is_train=True):

    if backbone_network.lower() == 'darknet':
        net = darknet.RT(n_classes=19, pretrained=is_train ,PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_DarkNET19)
    elif backbone_network.lower() == 'shufflenet':
        net = shufflenet.RT(n_classes=19, pretrained=is_train, PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_SHUFFLENET)
    elif backbone_network.lower() == 'mobilenet':
        net = mobilenet.RT(n_classes=19 ,pretrained=is_train, PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_MOBILENET)
    else:
        raise NotImplementedError

    if modelpath is not None:
        net.load_state_dict(torch.load(modelpath))

    print("Using LiteSeg with" ,backbone_network)
    return net


class LiteSeg(nn.Module):

    def __init__(self, num_classes=19, pretrained=True, PRETRAINED_WEIGHTS="."):

        super(LiteSeg, self).__init__()
        print("LiteSeg-DarkNet...")
        if pretrained:
            self.resnet_features = darknet.Darknet19(weights_file=PRETRAINED_WEIGHTS)
        else:
            self.resnet_features = darknet.Darknet19(weights_file=None)
        rates = [1, 3, 6, 9]

        self.aspp1 = ASPP(1024, 96, rate=rates[0])
        self.aspp2 = ASPP(1024, 96, rate=rates[1])
        self.aspp3 = ASPP(1024, 96, rate=rates[2])
        self.aspp4 = ASPP(1024, 96, rate=rates[3])

        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1024, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())

        # self.conv1 = nn.Conv2d(1504, 96, 1, bias=False)#480 1504
        self.conv1 = SeparableConv2d(1504, 96, 1)
        self.bn1 = nn.BatchNorm2d(96)

        # adopt [1x1, 48] for channel reduction.
        # self.conv2 = nn.Conv2d(128, 32, 1, bias=False)#128 for no previous feature 7  ,64 ---3
        self.conv2 = SeparableConv2d(128, 32, 1)
        self.bn2 = nn.BatchNorm2d(32)
        #
        self.last_conv = nn.Sequential(  # nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1, bias=False),
            SeparableConv2d(128, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            SeparableConv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, num_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        # print('x ',x.size(),' low features',low_level_features.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # print('x1',x1.size())
        x = torch.cat((x, x1, x2, x3, x4, x5), dim=1)  # x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)

        # ablation=torch.max(x, 1)[1]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),  ##/4 --7  /2 ---3
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)
        # ablation=torch.max(x, 1)[1]
        # print('x',x.size())

        ##comment to remove low feature
        # print('low level features',low_level_features.size())
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)

        # ablation=torch.max(x, 1)[1]
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x  # ,ablation

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ASPP(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate = rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            # self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            self.conv1 = SeparableConv2d(planes, planes, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()

            # self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes, planes, kernel_size, 1, padding, rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        # x = self.relu(x)
        if self.rate != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
