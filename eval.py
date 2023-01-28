from dataset.CamVid import CamVid
import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np

from model.builder_model import build_model
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou, colour_code_segmentation
import tqdm


def eval(model, dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data)[0].squeeze()  ### torch.Size([3, 512, 640]) ok
            predict = reverse_one_hot(predict)  ### torch.Size([512, 640])
            predict = np.array(predict)   #### 327680
            # predict = colour_code_segmentation(np.array(predict), label_info)

            label = label.squeeze()  ###torch.Size([4, 512, 640])
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)
            # label = np.array(label)[:1]    ###(4, 512, 640)
            # label = colour_code_segmentation(np.array(label), label_info)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou_list = per_class_iu(hist)[:-1]   ######len(2,)
        miou_list = per_class_iu(hist)  #######3
        miou_dict, miou = cal_miou(miou_list, csv_path)
        print('IoU for each class:')
        for key in miou_dict:
            print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="LMFFNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='D:/Code/Data/UAVdata/', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet18", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=3, help='num of object classes (with void)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='focalloss', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # create dataset and dataloader
    test_path = os.path.join(args.data, 'test')
    # test_path = os.path.join(args.data, 'train')
    test_label_path = os.path.join(args.data, 'test_labels')
    # test_label_path = os.path.join(args.data, 'train_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset = CamVid(test_path, test_label_path,  csv_path,
        scale=(args.crop_height, args.crop_width),loss=args.loss, mode='test')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # model = BiSeNet(args.num_classes, args.context_path)
    model = build_model(args.model, num_classes=args.num_classes)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.model+'_'+args.optimizer+'_latest_loss.pth')
    # args.checkpoint_path = os.path.join(args.checkpoint_path,args.model + '_' + args.optimizer + '_latest_loss.pth')
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')


    # get label info
    # label_info = get_label_info(csv_path)
    # test
    eval(model, dataloader, args, csv_path)


if __name__ == '__main__':
    params = [
        # '--checkpoint_path', 'path/to/ckpt',
        '--model', "FCNs",  # LMFFNet、BiSeNet、STDCNet、SSFPN
        '--checkpoint_path', 'checkpoints/',
        '--data', 'D:/Code/Data/uavdata/',
        # '--data', 'D:/Code/Data/uav_small/',
        '--cuda', '0',
        '--context_path', 'resnet18',
        '--num_classes', '3',
        '--optimizer', 'sgd',
        '--loss', 'focalloss',  # loss function,focalloss , dice or crossentropy'

    ]
    main(params)
