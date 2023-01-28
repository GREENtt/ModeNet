import sys
import argparse
import torch

from metrix.flops_count import get_model_complexity_info
from model.AGNet import AGNet
from model.BiSeNetV2 import BiSeNetV2
from model.builder_model import build_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    with torch.cuda.device(args.device):
        net = build_model(args.model,num_classes=3).cuda()

        flops, params = get_model_complexity_info(net, (3, 512, 640),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  ost=ost)
        print('Flops: ' + flops)
        print('Params: ' + params)