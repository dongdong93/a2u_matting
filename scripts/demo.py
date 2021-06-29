import os
import cv2
from time import time
from PIL import Image

import torch
import torch.nn as nn
import numpy as np

from resnet import resnetmat
from utils import *

import argparse
from config import cfg

IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

backbone = {
    'resnet34': resnetmat,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deep-Image-Matting")
    parser.add_argument("--cfg",
        default="./config/adobe.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,)
    parser.add_argument("--img_path", type=str, default="./demo/1.png", help="path to load the test image")
    parser.add_argument("--trimap_path", type=str, default="./demo/1_trimap.png", help="path to load the trimap")
    parser.add_argument("--out_path", type=str, default="./demo/1_result.png", help="path to save the test result")
    parser.add_argument("--model_path", type=str, default="./model/a2u_matting.pth", help="path to load the pretrained model")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()

def main():
    args = get_arguments()
    cfg.merge_from_file(args.cfg)

    # for distributed training
    torch.cuda.set_device(args.local_rank)

    try:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    except:
        args.distributed = False
        args.world_size = 1

    if not torch.distributed.is_initialized():
        if args.distributed:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        else:
            torch.distributed.init_process_group(backend='gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    # instantiate network
    net_type = backbone[cfg.MODEL.backbone]
    if cfg.MODEL.backbone == 'resnet34':
        net = net_type(
            conf=cfg,
            pretrained=True,
            distribute=None
        )
    else:
        raise NotImplementedError("backbone does not exist")
    net.to(device)
    net = SingleGPU(net)

    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict'])

    demo(net, args.img_path, args.trimap_path, args.out_path)

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

def demo(net, img_path, trimap_path, out_path):
    with torch.no_grad():
        net.eval()
        image, trimap = read_image(img_path).astype('float32'), read_image(trimap_path)
        image = (IMG_SCALE * image - IMG_MEAN) / IMG_STD
        if len(trimap.shape)==3:
            trimap = trimap.mean(2)
        trimap[(trimap!=0) & (trimap!=255)] = 128
        trimap = np.expand_dims(trimap, axis=2)
        image = np.concatenate((image, trimap/255.), axis=2)

        h, w = image.shape[:2]

        imsize = np.asarray(image.shape[:2], dtype=np.float)
        newsize = np.ceil(imsize / cfg.MODEL.stride) * cfg.MODEL.stride
        padh = int(newsize[0]) - h
        padw = int(newsize[1]) - w
        image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode="reflect")
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0).astype('float32'))
        inputs = inputs.to(device)

        torch.cuda.synchronize()
        start = time()
        outputs = net(inputs)
        torch.cuda.synchronize()
        end = time()

        outputs = outputs.squeeze().cpu().numpy()

        alpha = outputs[:h, :w]
        alpha = np.clip(alpha, 0, 1) * 255.

        trimap = trimap.squeeze().astype(np.uint8)
        mask = np.equal(trimap, 128).astype('float32')

        alpha = (1 - mask) * trimap + mask * alpha

        Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)).save(out_path)

        running_frame_rate = 1 * float(1 / (end - start))
        print('framerate: {0:.2f}Hz'.format(running_frame_rate))


if __name__ == '__main__':
    main()
