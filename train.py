import argparse

import torchvision
from PIL.ImageQt import rgb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from dataset.CamVid import CamVid
from model.build_BiSeNet import BiSeNet
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np

from model.builder_model import build_model
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss, focal_loss,CrossEntropyLoss2d


def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()   ###torch.Size([1, 3, 512, 640])
                label = label.cuda()   ###torch.Size([1,512, 640])

            # get RGB predict image
            if args.model == 'OURS':
                output = model(data, label)
            else:
                output = model(data)
            if type(output) is tuple:
                output = output[0]

            predict = output.squeeze() ###torch.Size([3, 224, 224])
            predict = reverse_one_hot(predict) ###torch.Size([224, 224])
            predict = np.array(predict)

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)
            # print('!!!!!!!!!!!',label.size())

            # compute per pixel accuracy

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.4f' % precision)
        print('mIoU for validation: %.4f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    ###-----GREENT------------------------------------------
    # resume = True
    #
    # # if resume:
    # if os.path.isfile(os.path.join(args.save_model_path, 'latest_dice_loss.pth')) or os.path.join(
    #         args.save_model_path, 'latest_dice_loss.pth'):
    #     print("Resume from checkpoint...")
    #     checkpoint = torch.load(os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
    #     model.module.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     initepoch = checkpoint['epoch']
    #     print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
    # else:
    #     print("====>no checkpoint found.")
    #     initepoch = 0
    ###----------GREENT-------------------------------------------

    writer = SummaryWriter(comment=''.format(args.optimizer))

    # writer = SummaryWriter("./runs/logs")
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        # loss_func = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,500,500])).float()).cuda()
        loss_func = CrossEntropyLoss2d(weight=torch.from_numpy(np.array([1, 10, 10])).float()).cuda()
    elif args.loss == 'focalloss':
        loss_func = focal_loss().cuda()

    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):   ### GREENT添加了initepoch参数
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        with torch.no_grad():
            model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        # tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (image, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()   ###torch.Size([8, 3, 512, 640])  float32
                label = label.cuda()  ###torch.Size([8, 512, 640])  int64
            # print("epoch：", epoch, "的第", i, "个inputs", data.data.size(), "labels", label.data)
            # print(len(label), image.size(), label.size())
            if args.model == 'OURS':
                output = model(image, label)
            else:
                output = model(image)
            # print(image.size(), output.size(), label.size())   ###torch.Size([8, 3, 512, 640]) torch.Size([8, 3, 512, 640]) torch.Size([8, 512, 640])
            if type(output) is tuple:
                loss = 0
                for i in range(len(output)):
                    loss = loss+loss_func(output[i], label)
            else:
                loss = loss_func(output, label)

            # if args.model=='SSFPN':
            #     output, output_sup1, output_sup2, output_sup3, output_sup4 = model(image)
            #     loss1 = loss_func(output, label)
            #     loss2 = loss_func(output_sup1, label)
            #     loss3 = loss_func(output_sup2, label)
            #     loss4 = loss_func(output_sup3, label)
            #     loss5 = loss_func(output_sup4, label)
            #     loss = loss1 + loss2 + loss3 + loss4 + loss5
            #
            # elif args.model == 'BiSeNet':
            #     output, output_sup1, output_sup2 = model(image)
            #     loss1 = loss_func(output, label)
            #     loss2 = loss_func(output_sup1, label)
            #     loss3 = loss_func(output_sup2, label)
            #     loss = loss1 + loss2 + loss3
            #
            # else:
            #     output = model(image)
            #     loss = loss_func(output, label)
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)

        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)


            ###------------------GREENT--------------------------
            # checkpoint = {"model_state_dict": model.module.state_dict(),
            #               "optimizer_state_dict": optimizer.state_dict(),
            #               "epoch": epoch}
            # checkpoint_path = os.path.join(args.save_model_path, 'latest_dice_loss.pth')
            # torch.save(checkpoint, checkpoint_path)   ####GREENT
            ###------------------GREENT--------------------------

            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, args.model+'_'+args.optimizer+'_latest_loss.pth'))

        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou

                ###------------------GREENT---------------------------
                # checkpoint = {"model_state_dict": model.state_dict(),
                #               "optimizer_state_dict": optimizer.state_dict(),
                #               "epoch": epoch}
                # checkpoint_path = os.path.join(args.save_model_path, '/best_dice_loss.pth')
                # torch.save(checkpoint, checkpoint_path)  ####GREENT
                ###------------------GREENT--------------------------


                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, args.model+'_best_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="STDCNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="camvid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=640,  ####720
                        help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=512,  ####960
                        help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet18",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='D:/Code/Data/UAVdata/', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers')  ########  4
    parser.add_argument('--num_classes', type=int, default=11,  ########  32
                        help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='focalloss', help='loss function,focalloss , dice or crossentropy')

    args = parser.parse_args(params)

    # create dataset and dataloader
    train_path = [os.path.join(args.data+'train'), os.path.join(args.data+'val')]
    train_label_path = [os.path.join(args.data+'train_labels'), os.path.join(args.data+'val_labels')]
    test_path = os.path.join(args.data+'test')
    test_label_path = os.path.join(args.data+'test_labels')
    csv_path = os.path.join(args.data+'class_dict.csv')
    ######数据增强####################################
    # train_transform = Compose([transforms.RandomRotate90(), transforms.Flip(),
    #                            OneOf([transforms.HueSaturationValue(),
    #                                   transforms.RandomBrightness(),
    #                                   transforms.RandomContrast(), ], p=1),  # 按照归一化的概率选择执行哪一个
    #                            transforms.Resize(args.crop_height, args.crop_width),
    #                            transforms.Normalize(), ])
    # val_transform = Compose([transforms].Resize(args.crop_height, args.crop_width),
    #                         transforms.Normalize(), )

    ###########################################

    dataset_train = CamVid(train_path, train_label_path, csv_path,
                           scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,   #### True
        num_workers=args.num_workers,
        drop_last=False    #### True
    )

    dataset_val = CamVid(test_path, test_label_path, csv_path,
                         scale=(args.crop_height, args.crop_width),
                         loss=args.loss, mode='test')

    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )


    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # model = BiSeNet(args.num_classes, args.context_path)
    model = build_model(args.model, num_classes=args.num_classes)
    # print(model)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model)

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

    # val(args, model, dataloader_val, csv_path)



if __name__ == '__main__':
    params = [
        '--model', "OURS", # LMFFNet、BiSeNet、STDCNet、SSFPN、AGNet
        '--num_epochs', '10',  # 1000
        '--learning_rate', '2.5e-2',  ###2.5e-2
        '--data', 'D:/Code/Data/uavdata/',
        '--num_workers', '0',  ###8
        '--num_classes', '3', ###12
        '--cuda', '0',
        '--batch_size', '1',  # 6 for resnet101, 12 for resnet18
        '--save_model_path', './checkpoints/',
        '--optimizer', 'sgd',
        '--loss', 'focalloss',  # loss function,focalloss , dice or crossentropy'

    ]
    main(params)

