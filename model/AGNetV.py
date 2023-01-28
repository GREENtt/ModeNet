# _*_ coding: utf-8 _*_
"""
Reference from: https://github.com/CoinCheung/BiSeNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

from torch.nn import BatchNorm2d, Conv2d
from tensorboardX import summary, SummaryWriter


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio   ###？？？？？？？
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.PReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.PReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio  ###？？？？？？？
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.PReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.PReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()
        # self.s5_5 = SpatialAttention()  ####

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class SCWAFusionNet(nn.Module):

    def __init__(self):
        super(SCWAFusionNet, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        ###### out = self.conv(left + right)
        ###### return out
        return left, right



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, num_classes):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(
                mid_chan, num_classes, kernel_size=1, stride=1,
                padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        if not size is None:
            feat = F.interpolate(feat, size=size,
                mode='bilinear', align_corners=True)
        return feat



class AGNet(nn.Module):

    def __init__(self, num_classes):
        super(AGNet, self).__init__()
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.pffn = SCWAFusionNet()

        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(3, 1024, num_classes)
        self.aux2 = SegmentHead(16, 128, num_classes)
        self.aux3 = SegmentHead(32, 128, num_classes)
        self.aux4 = SegmentHead(64, 128, num_classes)
        self.aux5_4 = SegmentHead(128, 128, num_classes)
        ###############################################
        self.apfn = PyrmidFusionNet(128, 4, 64, num_classes)

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x) ##torch.Size([4, 128, 64, 80])

        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_detail, high_seg = self.pffn(feat_d, feat_s)  ###torch.Size([4, 128, 64, 80])
        feat_head = self.apfn(high_seg, feat_detail )  ###torch.Size([4, 3, 64, 80])
        logits = self.head(feat_head, size)  ###torch.Size([6, 3, 512, 640])
        logits_aux2 = self.aux2(feat2, size)
        logits_aux3 = self.aux3(feat3, size)
        logits_aux4 = self.aux4(feat4, size)
        logits_aux5_4 = self.aux5_4(feat5_4, size)

        #################GREENT#########
        # out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        # out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
        # out = F.relu(self.gamma * out + (1 - self.gamma) * x)  ###torch.Size([4, 64, 64, 80])
        #################GREENT#########
        return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4

###########################################################################



class SpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        ## self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.fuse = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()   ###torch.Size([4, 64, 64, 80])

        mxpool = F.max_pool2d(x, [h, 1])  # .view(b,c,-1).permute(0,2,1) ###torch.Size([4, 64, 1, 80])
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)  ###torch.Size([4, 64, 1, 80])
        mxpool = self.bn_sh1(mxpool)   ###torch.Size([4, 64, 1, 80])

        avgpool = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        avgpool = F.conv2d(avgpool, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)   ###torch.Size([4, 64, 1, 80])

        att = torch.softmax(torch.mul(mxpool, avgpool), 1)  ###torch.Size([4, 64, 1, 80])
        attt1 = att[:, 0, :, :].unsqueeze(1)  ###torch.Size([4, 1, 1, 80])
        attt2 = att[:, 1, :, :].unsqueeze(1)  ### torch.Size([4, 1, 1, 80])

        fusion = attt1 * avgpool + attt2 * mxpool  ###torch.Size([4, 64, 1, 80])
        # out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        # out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
        # out = F.relu(self.gamma * out + (1 - self.gamma) * x)  ###torch.Size([4, 64, 64, 80])
        return fusion


class ChannelWise(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWise, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_pool(y)

        return x * y


class PyrmidFusionNet(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, num_classes=3):
        super(PyrmidFusionNet, self).__init__() ###128,4,64

        # self.lateral_low = ConvBNReLU(channels_low, channels_high, 1, 1, padding=0)
        self.lateral_low = ConvBNReLU(channels_high, channel_out, 1, 1, padding=0)

        self.conv_low = ConvBNReLU(channel_out, channel_out, 3, 1,  padding=1)

        self.sa = SpatialAttention(channel_out, channel_out)

        self.conv_high = ConvBNReLU(channels_high, channel_out, 3, 1, padding=1)

        self.ca = ChannelWise(channel_out)

        self.FRB = nn.Sequential(
             ConvBNReLU(channels_high, channels_low, 1, 1, padding=0),
             ConvBNReLU(channels_low, channel_out, 3, 1, padding=1, groups=1)
        )
        self.classifier = nn.Sequential(
             ConvBNReLU(channel_out, channel_out, 3, 1, padding=1, groups=1),
             nn.Dropout(p=0.15),
             Conv2d(channel_out, num_classes, 1, 1, padding=0))
        # self.conv = Conv2d(3, 128, 1, 1, padding=0)
        # self.apf = ConvBNReLU(channel_out, channel_out, 3, 1, padding=1, group=1)

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()
        lat_low = self.lateral_low(x_low)   ###torch.Size([4, 64, 64, 80])
        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)
        # x_high = self.lateral_low(high_up1)  ###torch.Size([4, 64, 64, 80])
        conv_high = self.conv_high(high_up1)  ###torch.Size([4, 64, 64, 80])

        concate = torch.cat([lat_low, conv_high], 1)  ###torch.Size([4, 128, 64, 80])

        concate = self.FRB(concate)   ###torch.Size([4, 64, 64, 80])

        conv_low = self.conv_low(lat_low)   ###torch.Size([4, 64, 64, 80])

        sa = self.sa(concate)  ###torch.Size([6, 64, 64, 80])

        ca = self.ca(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        att_out = mul1 + mul2   ###torch.Size([6, 64, 64, 80])

        sup = self.classifier(att_out)   ###torch.Size([4, 3, 64, 80])
        # sup = self.conv(sup)  ###torch.Size([4, 64, 64, 80])
        # APF = self.apf(att_out)
        # return APF, sup
        return sup



"""print layers and params of network"""
if __name__ == '__main__':
    x = torch.randn(8, 3, 512, 640)
    model = AGNet(num_classes=3)
    # print(model)
    # summary(model, torch.rand((2, 3, 512, 640)))
    # writer = SummaryWriter('log')
    # writer.add_graph(model, x)
    # writer.close()
    print(model)
    logits = model(x)[0]
    print(logits.size())

