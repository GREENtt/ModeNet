import cv2
import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np

from model.builder_model import build_model
from utils import reverse_one_hot, get_label_info, colour_code_segmentation


def predict_on_image(model, args):
    # pre-processing on image
    for i in os.listdir(args.data):
        file_path = args.data + i
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(i)
        # image = Image.fromarray(image).convert('RGB')
        resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
        resize_det = resize.to_deterministic()
        image = resize_det.augment_image(image)
        image = Image.fromarray(image).convert('RGB')
        image = transforms.ToTensor()(image)   ### torch.Size([3, 512, 640])
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
        # read csv label path
        label_info = get_label_info(args.csv_path)
        # predict
        model.eval()
        predict = model(image)[0].squeeze()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict), label_info)  ###(512, 640, 3)
        #### predict = cv2.resize(np.uint8(predict), (960, 720))
        predict = cv2.resize(np.uint8(predict), (640, 512))  # (640, 512, 3)
        cv2.imwrite(os.path.join(args.save_path, i), cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="resnet18", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet18", help='The context path model you are using.')
    parser.add_argument('--num_classes', type=int, default=3,  #### 12
                        help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='focalloss', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # model = BiSeNet(args.num_classes, args.context_path)
    model = build_model(args.model, num_classes=args.num_classes)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.model+'_'+args.optimizer+'_latest_loss.pth')
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # predict on image
    if args.image:
        predict_on_image(model, args)

    # predict on video
    if args.video:
        pass


if __name__ == '__main__':
    params = [
        '--image',
        '--model', "FCNs",  # LMFFNet、BiSeNet、STDCNet、SSFPN、AGNet
        '--data', 'D:/Code/Data/uavdata/test/',
        # '--checkpoint_path', '/path/to/ckpt',
        '--checkpoint_path', 'checkpoints/',
        '--cuda', '0',
        '--csv_path', 'D:/Code/Data/uavdata/class_dict.csv',
        '--save_path', './img/FCNs/',
        '--optimizer', 'sgd',
        '--loss', 'focalloss',  # loss function,focalloss , dice or crossentropy'
    ]
    main(params)
