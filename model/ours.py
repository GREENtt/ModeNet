import math

import torch
from PIL.Image import Image
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Softmax

from model.base_model.resnet import ResNet, resnet18



def get_norm(name, nc):
    if name == 'batchnorm':
        return nn.BatchNorm2d(nc)
    if name == 'instancenorm':
        return nn.InstanceNorm2d(nc)
    raise ValueError('Unsupported normalization layer: {:s}'.format(name))


def get_nonlinear(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError('Unsupported activation layer: {:s}'.format(name))


def conv2d(c_in, c_out, k_size=3, stride=2, pad=1, dilation=1, bn=True, lrelu=True, leak=0.2):
    layers = []
    if lrelu:
        layers.append(nn.LeakyReLU(leak))
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def fc(input_size, output_size):
    return nn.Linear(input_size, output_size)


def embedding_lookup(embeddings, embedding_ids, GPU=False):
    batch_size = len(embedding_ids)
    embedding_dim = embeddings.shape[3]
    local_embeddings = []
    for id_ in embedding_ids:
        if GPU:
            local_embeddings.append(embeddings[id_].cpu().numpy())
        else:
            local_embeddings.append(embeddings[id_].data.numpy())
    local_embeddings = torch.from_numpy(np.array(local_embeddings))
    if GPU:
        local_embeddings = local_embeddings.cuda()
    local_embeddings = local_embeddings.reshape(batch_size, embedding_dim, 1, 1)
    return local_embeddings


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv1(input)
        return self.bn(x)


class ExtraConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)  ###torch.Size([1, 64, 256, 320])
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(images, En, De, embeddings, embedding_ids, GPU=False, encode_layers=False):
        encoded_source, encode_layers = En(images)
        local_embeddings = embedding_lookup(embeddings, embedding_ids, GPU=GPU)
        if GPU:
            encoded_source = encoded_source.cuda()
            local_embeddings = local_embeddings.cuda()
        embedded = torch.cat((encoded_source, local_embeddings), 1)
        fake_target = De(embedded, encode_layers)
        if encode_layers:
            return fake_target, encoded_source, encode_layers
        else:
            return fake_target,


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_dim=2, disc_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv2d(img_dim, disc_dim, bn=False)
        self.conv2 = conv2d(disc_dim, disc_dim * 2)
        self.conv3 = conv2d(disc_dim * 2, disc_dim * 4)
        self.conv4 = conv2d(disc_dim * 4, disc_dim * 8)
        self.fc1 = fc(disc_dim * 8 * 8 * 8, 1)
        self.fc2 = fc(disc_dim * 8 * 8 * 8, num_classes)

    def forward(self, images):
        batch_size = images.shape[0]
        h1 = self.conv1(images)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        tf_loss_logit = self.fc1(h4.reshape(batch_size, -1))
        tf_loss = torch.sigmoid(tf_loss_logit)
        cat_loss = self.fc2(h4.reshape(batch_size, -1))
        return tf_loss, tf_loss_logit, cat_loss


class ResBlk(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResBlk, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, 3, 1, 1),
            get_norm('batchnorm', n_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 3, 1, 1),
            get_norm('batchnorm', n_out),
        )

    def forward(self, x):
        return self.layers(x) + x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _Generator(nn.Module):
    def __init__(self, input_channels, output_channels, last_nonlinear):
        super(_Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 7, 1, 3),  # n_in, n_out, kernel_size, stride, padding
            get_norm('instancenorm', 32),
            get_nonlinear('relu'),
            nn.Conv2d(32, 64, 4, 2, 1),
            get_norm('instancenorm', 64),
            get_nonlinear('relu'),
            nn.Conv2d(64, 128, 4, 2, 1),
            get_norm('instancenorm', 128),
            get_nonlinear('relu'),
            nn.Conv2d(128, 256, 4, 2, 1),
            get_norm('instancenorm', 256),
            get_nonlinear('relu'),
        )
        self.resblk = nn.Sequential(
            ResBlk(256, 256),
            ResBlk(256, 256),
            ResBlk(256, 256),
            ResBlk(256, 256),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            get_norm('instancenorm', 128),
            get_nonlinear('relu'),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            get_norm('instancenorm', 64),
            get_nonlinear('relu'),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            get_norm('instancenorm', 32),
            get_nonlinear('relu'),
            nn.ConvTranspose2d(32, output_channels, 7, 1, 3),
            get_nonlinear(last_nonlinear),
        )

    def forward(self, x, a=None):
        if a is not None:
            assert a.dim() == 2 and x.size(0) == a.size(0)
            a = a.type(x.dtype)
            a = a.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat((x, a), dim=1)
        h = self.conv(x)
        h = self.resblk(h)
        y = self.deconv(h)
        return y


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.AMN = _Generator(4, 3, 'tanh')
        self.SAN = _Generator(3, 1, 'sigmoid')

    def forward(self, x, a):
        y = self.AMN(x, a)
        m = self.SAN(x)
        y_ = y * m + x * (1 - m)
        return y_, m


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(32, 64, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(64, 128, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(128, 256, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(256, 512, 4, 2, 1),
            get_nonlinear('lrelu'),
            nn.Conv2d(512, 1024, 4, 2, 1),
            get_nonlinear('lrelu'),
        )
        self.src = nn.Conv2d(1024, 1, 3, 1, 1)
        self.cls = nn.Sequential(
            nn.Conv2d(1024, 1, 2, 1, 0),
            get_nonlinear('sigmoid'),
        )

    def forward(self, x):
        h = self.conv(x)
        return self.src(h), self.cls(h).squeeze().unsqueeze(1)


class CrossAttentionModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(CrossAttentionModel, self).__init__()

        ########################
        self.encoder1 = nn.Linear(20480, 128)
        self.encoder2 = nn.Linear(20480, 128)

        # self.affine_a = nn.Linear(8, 8, bias=False)
        # self.affine_v = nn.Linear(8, 8, bias=False)
        self.affine_a = nn.Linear(3, 3, bias=False)
        self.affine_v = nn.Linear(3, 3, bias=False)

        # self.W_a = nn.Linear(8, 32, bias=False)
        # self.W_v = nn.Linear(8, 32, bias=False)
        self.W_a = nn.Linear(3, 32, bias=False)
        self.W_v = nn.Linear(3, 32, bias=False)
        self.W_ca = nn.Linear(256, 32, bias=False)
        self.W_cv = nn.Linear(256, 32, bias=False)

        self.W_ha = nn.Linear(32, 3, bias=False)
        self.W_hv = nn.Linear(32, 3, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.regressor = nn.Sequential(nn.Linear(640, 128),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, num_classes))

    #def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm):
        #f1 = f1.squeeze(1)
        #f2 = f2.squeeze(1)

        f1_norm = torch.nn.functional.interpolate(f1_norm.float(), scale_factor=1/4)
        f2_norm = torch.nn.functional.interpolate(f2_norm.unsqueeze(1).float(), scale_factor=1/4)
        # print(f1_norm.size())   ###torch.Size([1, 3, 640, 512])

        #f1_norm = F.normalize(f1_norm, p=2, dim=2, eps=1e-12)
        #f2_norm = F.normalize(f2_norm, p=2, dim=2, eps=1e-12)
        b, c, h, w = f1_norm.size()
        f1_norm = f1_norm.reshape(b, c, h*w)  ##torch.Size([1, 1, 640, 512])
        f2_norm = f2_norm.reshape(b, 1, h*w)  ##torch.Size([1, 1, 640, 512])
        f2_norm = torch.cat([f2_norm,f2_norm,f2_norm],dim=1)
        fin_audio_features = []
        fin_visual_features = []
        sequence_outs = []

        for i in range(f1_norm.shape[0]):
            audfts = f1_norm[i, :, :]#.transpose(0, 1)   ###torch.Size([3, 640, 512])
            visfts = f2_norm[i, :, :]#.transpose(0, 1)   ###torch.Size([1, 640, 512])

            aud_fts = self.encoder1(audfts)  ###torch.Size([3, 640, 128])
            vis_fts = self.encoder2(visfts.float())  ##torch.Size([1, 640, 128])

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)  #torch.Size([4, 640, 128])
            a_t = self.affine_a(aud_vis_fts.transpose(0, 1))  ##torch.Size([640, 4, 128])
            att_aud = torch.mm(aud_fts.transpose(0, 1), a_t.transpose(0, 1))
            # mm (128,4)*(4,128) (128,128)
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1]))) ##torch.Size([128, 256])

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)  ##torch.Size([3, 256])
            v_t = self.affine_v(aud_vis_fts.transpose(0, 1))  ##torch.Size([256, 3])
            att_vis = torch.mm(vis_fts.transpose(0, 1), v_t.transpose(0, 1))   ##torch.Size([128, 256])
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1]))) ##torch.Size([128, 256])

            H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts.transpose(0, 1)))  ##torch.Size([128, 32])
            H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts.transpose(0, 1)))  ##torch.Size([128, 32])

            att_audio_features = self.W_ha(H_a).transpose(0, 1) + aud_fts   ##torch.Size([3, 128])
            att_visual_features = self.W_hv(H_v).transpose(0, 1) + vis_fts   ##torch.Size([3, 128])

            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), 1)  ##torch.Size([3, 256])
            # outs = self.regressor(audiovisualfeatures) #.transpose(0,1))  ##
            #seq_outs, _ = torch.max(outs,0)
            #print(seq_outs)
            sequence_outs.append(audiovisualfeatures)
            fin_audio_features.append(att_audio_features)
            fin_visual_features.append(att_visual_features)
        final_aud_feat = torch.stack(fin_audio_features)
        final_vis_feat = torch.stack(fin_visual_features)
        final_outs = torch.stack(sequence_outs)  ##torch.Size([1, 3, 256])
        return final_outs #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)


class FeatureFusionModel(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusionModel, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class OURS(torch.nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d,):
        super(OURS, self).__init__()
        # self.baseline = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

        self.in_dim = 64
        self.baseline = resnet18(num_classes=num_classes)
        conv_channel = 128
        '''######################### '''
        self.attblock = CrossAttentionModel(num_classes)
        self.extra_conv = ExtraConv(3, 64, 7, stride=1, padding=3)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.ffm = FeatureFusionModel(conv_channel * 2, conv_channel * 2, 1, norm_layer)

    def forward(self, image, target):
        # print(image.size(), target.size()) #torch.Size([2, 3, 640, 512]) torch.Size([2, 640, 512])
        ori_out = self.baseline(image)
        extra_conv = self.extra_conv(ori_out)  ###torch.Size([2, 32, 318, 254]
        # print('111', ori_out.size(), target.size(), extra_conv.size())
        OP = 1-self.attblock(ori_out, target)

        fakeOP = self.generator(extra_conv)
        discriminator = self.discriminator(OP, fakeOP)
        out = self.ffm(fakeOP, ori_out)
        return out, discriminator,
        # return ori_out



