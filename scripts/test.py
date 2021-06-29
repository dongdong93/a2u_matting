import os
import cv2
from time import time
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnetmat
from data_generator import DataGenerator
from utils import *

import argparse
from config import cfg

backbone = {
    'resnet34': resnetmat,
}

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
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()

def main():
    args = get_arguments()
    cfg.merge_from_file(args.cfg)

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
    net.cuda()
    net = SingleGPU(net)

    RESULT_DIR = os.path.join(cfg.TEST.result_dir, cfg.EXP)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    start_epoch = 0
    net.num_update = 0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'sad': [],
        'mse': [],
        'grad': [],
        'conn': []
    }

    RESTORE_FROM = os.path.join(cfg.TRAIN.model_save_dir, cfg.DATASET.name.lower(), cfg.EXP, cfg.TEST.checkpoint)
    if os.path.isfile(RESTORE_FROM):
        # checkpoint = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(RESTORE_FROM)
        net.load_state_dict(checkpoint['state_dict'])
        print("==> load checkpoint '{}'"
                .format(RESTORE_FROM))


    dataset = DataGenerator
    testset = dataset(cfg, phase='test', test_scale='origin')


    test_loader = DataLoader(
        testset,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=None
    )

    image_list = [name.split('\t') for name in open(cfg.DATASET.val_list).read().splitlines()]
    # switch to eval mode
    net.eval()

    with torch.no_grad():
        sad = []
        mse = []
        grad = []
        conn = []
        avg_frame_rate = 0
        start = time()
        for i, sample in enumerate(test_loader):
            sample = testset.__getitem__(i)
            image, target = sample['image'], sample['alpha']
            image = torch.unsqueeze(image, dim=0)
            target = torch.unsqueeze(target, dim=0)


            h, w = image.size()[2:]
            image = image.squeeze().numpy().transpose(1, 2, 0)
            imsize = np.asarray(image.shape[:2], dtype=np.float)
            newsize = np.ceil(imsize / cfg.MODEL.stride) * cfg.MODEL.stride
            padh = int(newsize[0]) - h
            padw = int(newsize[1]) - w
            image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode="reflect")
            inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))

            # inference
            inputs = inputs.cuda()
            try:
                outputs = net(inputs).squeeze().cpu().numpy()
            except:
                continue

            end = time()

            alpha = outputs[:h, :w]
            alpha = np.clip(alpha, 0, 1) * 255.

            trimap = target[:, 1, :, :].squeeze().numpy()
            mask = np.equal(trimap, 128).astype(np.float32)

            alpha = (1 - mask) * trimap + mask * alpha
            gt_alpha = target[:, 0, :, :].squeeze().numpy() * 255.

            alpha.astype(np.uint8)
            gt_alpha.astype(np.uint8)

            path, image_name = os.path.split(image_list[i][0])
            Image.fromarray(alpha.astype(np.uint8)).save(os.path.join(RESULT_DIR, image_name))


            sad.append(compute_sad_loss(alpha, gt_alpha, mask))
            mse.append(compute_mse_loss(alpha, gt_alpha, mask))
            grad.append(compute_gradient_loss(alpha, gt_alpha, mask))
            conn.append(compute_connectivity_loss(alpha, gt_alpha, mask))

            running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
            avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
            print(
                'epoch: {0}, test: {1}/{2}, sad: {3:.2f}, SAD: {4:.2f}, MSE: {5:.4f},'
                'Grad: {6:.2f}, Conn: {7:.2f}, frame: {8:.2f}Hz/{9:.2f}Hz'
                .format(0, i+1, len(test_loader), sad[-1], np.mean(sad), np.mean(mse),
                np.mean(grad), np.mean(conn), running_frame_rate, avg_frame_rate)
            )
            start = time()

if __name__== '__main__':
    main()
