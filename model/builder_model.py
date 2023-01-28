from torchvision.models.segmentation import FCN

from model.AGNet import AGNet
from model.BiSeNetV import BiSeNetV
from model.BiSeNetV2 import BiSeNetV2
from model.DDRNet import DDRNet
from model.FCN import VGGNet, FCNs
from model.FastSCNN import FastSCNN
from model.LMFFNet import LMFFNet
from model.SPFNet import SPFNet
from model.SSFPN import SSFPN
from model.STDCNet import STDCNet
from model.UNet import UNet
from model.base_model.resnet import resnet18
from model.build_BiSeNet import BiSeNet
from model.green import green
from model.ours import OURS
from model.xception import xception, Xception

resnet18


def build_model(model_name, num_classes):

    if model_name == 'BiSeNet':
        return BiSeNet(num_classes=num_classes)
    if model_name == 'BiSeNetV':
        return BiSeNetV(num_classes=num_classes)
    if model_name == 'BiSeNetV2':
        return BiSeNetV2(num_classes=num_classes)
    # if model_name == 'LiteSeg':
    #     return LiteSeg(num_classes=num_classes)
    if model_name == 'STDCNet':
        return STDCNet(backbone='STDCNet813', n_classes=num_classes)
    if model_name == 'SPFNet':
        return SPFNet(classes=num_classes)
    if model_name == 'LMFFNet':
        return LMFFNet(num_classes=num_classes)
    if model_name == 'SSFPN':
        return SSFPN("resnet18", classes=num_classes)
    # if model_name == 'LiteSeg':
    #     return LiteSeg.build(backbone_network,None,CONFIG,is_train=True)
    if model_name == 'AGNet':
        return AGNet(num_classes=num_classes)
    if model_name == 'resnet18':
        return resnet18(num_classes=num_classes)
    if model_name =='DDRNet':
        return DDRNet(num_classes=num_classes)
    if model_name =='green':
        return green(num_classes=num_classes)
    if model_name =='UNet':
        return UNet(num_classes=num_classes)
    if model_name =='FastSCNN':
        return FastSCNN(num_classes=num_classes)
    if model_name =='FCNs':
        return FCNs(num_classes=num_classes)
    if model_name =='OURS':
        return OURS(num_classes=num_classes)
    if model_name =='xception':
        return Xception(num_classes=num_classes)
