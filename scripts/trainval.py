import sys
import os

import argparse
from time import time

import cv2 as cv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from utils import *
import torch.backends.cudnn as cudnn
from data_generator import DataGenerator
from prefetcher import Prefetcher
from resnet import resnetmat
from inplace_abn import InPlaceABN
from functools import partial
from config import inplace_config as config
from config import cfg
import torch.multiprocessing as mp

cudnn.enabled = True


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
    rank = 0 if not args.distributed else torch.distributed.get_rank()

    # seeding for reproducbility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.TRAIN.random_seed)
    torch.manual_seed(cfg.TRAIN.random_seed)
    np.random.seed(cfg.TRAIN.random_seed)

    snapshot_dir = os.path.join(cfg.TRAIN.model_save_dir, cfg.DATASET.name.lower(), cfg.EXP)
    cfg.TRAIN.result_dir = os.path.join(cfg.TRAIN.result_dir, cfg.EXP)
    if rank == 0:
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        if not os.path.exists(cfg.TRAIN.result_dir):
            os.makedirs(cfg.TRAIN.result_dir)
        print("{}".format(cfg))

    # instantiate network
    net_type = backbone[cfg.MODEL.backbone]
    if cfg.MODEL.backbone == 'resnet34':
        net = net_type(
            conf=cfg,
            pretrained=True,
            distribute=args.distributed
        )
    else:
        raise NotImplementedError("backbone does not exist")

    net.cuda()

    # filter parameters
    pretrained_params = []
    learning_params = []
    for p in net.named_parameters():
        if 'dconv' in p[0] or 'pred' in p[0] or 'short' in p[0] or 'bconv' in p[0] or 'bilinear' in p[0]:
            learning_params.append(p[1])
        else:
            pretrained_params.append(p[1])

    # define optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': learning_params},
            {'params': pretrained_params, 'lr': cfg.TRAIN.initial_lr / cfg.TRAIN.mult},
        ],
        lr=cfg.TRAIN.initial_lr
    )

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)
    else:
        net = SingleGPU(net)

    if rank==0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # restore parameters
    start_epoch = 0
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


    cfg.TRAIN.restore = os.path.join(cfg.TRAIN.model_save_dir, cfg.DATASET.name.lower(), cfg.EXP, cfg.TRAIN.restore)
    if cfg.TRAIN.restore is not None:
        if os.path.isfile(cfg.TRAIN.restore):
            checkpoint = torch.load(cfg.TRAIN.restore)
            net.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'train_loss' in checkpoint:
                net.train_loss = checkpoint['train_loss']
            if 'val_loss' in checkpoint:
                net.val_loss = checkpoint['val_loss']
            if 'measure' in checkpoint:
                net.measure = checkpoint['measure']
            if rank==0:
                print("==> load checkpoint '{}' (epoch {})"
                      .format(cfg.TRAIN.restore, start_epoch))
        else:
            if rank==0:
                with open(os.path.join(cfg.TRAIN.result_dir, cfg.EXP + '.txt'), 'a') as f:
                    print("{}".format(cfg), file=f)
                print("==> no checkpoint found at '{}'".format(cfg.TRAIN.restore))

    dataset = DataGenerator
    if not cfg.TRAIN.evaluate_only:
        trainset = dataset(cfg, phase='train', test_scale='origin', crop_size=cfg.TRAIN.crop_size)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        else:
            train_sampler = None

        train_loader = DataLoader(
            trainset,
            batch_size=cfg.TRAIN.batch_size // args.world_size,
            shuffle=(train_sampler is None),
            num_workers=cfg.TRAIN.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler
        )
        train_loader = Prefetcher(train_loader)

    valset = dataset(cfg, phase='test', test_scale='origin')
    if args.distributed:
        test_sampler = TestDistributedSampler(valset)
    else:
        test_sampler = None

    val_loader = DataLoader(
        valset,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=test_sampler
    )

    if cfg.TRAIN.evaluate_only:
        validate(net, val_loader, start_epoch + 1, cfg)
        return



    resume_epoch = -1 if start_epoch == 0 else start_epoch
    scheduler = MultiStepLR(optimizer, milestones=[20, 26], gamma=0.1, last_epoch=resume_epoch)
    for epoch in range(start_epoch, cfg.TRAIN.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train
        train(net, train_loader, optimizer, epoch + 1, scheduler, cfg)
        scheduler.step()
        # val
        validate(net, val_loader, epoch + 1, cfg)

        if rank == 0:
            # save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': net.train_loss,
                'val_loss': net.val_loss,
                'measure': net.measure,
            }
            save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth')
            print(cfg.EXP + ' epoch {} finished!'.format(epoch + 1))
            if len(net.measure['sad']) > 1 and net.measure['sad'][-1] <= min(net.measure['sad'][:-1]):
                save_checkpoint(state, snapshot_dir, filename='model_best.pth')

        # reset the scheduler
    print('Experiments with ' + cfg.EXP + ' done!')




def train(net, train_loader, optimizer, epoch, scheduler, cfg):
    # switch to train mode
    net.train()
    cudnn.benchmark = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    running_loss = 0.0
    running_time = 0.0
    avg_frame_rate = 0.0
    # torch.cuda.synchronize()
    start = time()
    # create loader iterator
    iterator_train = iter(train_loader)
    for i in range(cfg.TRAIN.epoch_iterations):
        try:
            sample = next(iterator_train)
        except:
            iterator_train = iter(train_loader)
            sample = next(iterator_train)
        inputs, targets = sample['image'], sample['alpha']
        inputs, targets = inputs.cuda(), targets.cuda()
        # torch.cuda.synchronize()
        end_data = time()

        # forward
        outputs = net(inputs)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute loss
        loss = weighted_loss(outputs, targets)

        # backward + optimize
        loss.backward()
        optimizer.step()
        # collect and print statistics
        loss = loss.detach() * targets.shape[0]
        count = targets.new_tensor([targets.shape[0]], dtype=torch.long)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(loss, torch.distributed.ReduceOp.SUM)
        loss /= count.item()
        running_loss += loss.item()

        # torch.cuda.synchronize()
        end = time()

        running_frame_rate = cfg.TRAIN.batch_size * float(1 / (end - start))
        running_time += (end - start)
        avg_frame_rate = (avg_frame_rate * i + running_frame_rate) / (i + 1)
        if i % cfg.TRAIN.record_every == cfg.TRAIN.record_every - 1:
            net.train_loss['running_loss'].append(running_loss / (i + 1))
        if rank == 0:
            if i % cfg.TRAIN.print_every == cfg.TRAIN.print_every - 1:
                print('epoch: %d, train: %d/%d, lr: %.5f, '
                      'loss: %.5f, time: %fs/%fs, loaddata: %fs, lefttime: %fh, frame: %.2fHz/%.2fHz' % (
                          epoch,
                          i + 1,
                          cfg.TRAIN.epoch_iterations,
                          optimizer.param_groups[0]['lr'],
                          running_loss / (i + 1),
                          (end - start),
                          running_time / (i + 1),
                          (end_data - start),
                          ((cfg.TRAIN.num_epochs-epoch)*cfg.TRAIN.epoch_iterations+cfg.TRAIN.epoch_iterations-i-1)*(running_time / (i + 1))/3600,
                          running_frame_rate,
                          avg_frame_rate
                      ))
        start = time()
    net.train_loss['epoch_loss'].append(running_loss / (i + 1))


def validate(net, val_loader, epoch, cfg):
    # switch to eval mode
    net.eval()
    end = time()
    cudnn.benchmark = False

    image_list = [name.split('\t') for name in open(cfg.DATASET.val_list).read().splitlines()]

    # deal with remainder
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    all_reduce = partial(torch.distributed.all_reduce,
                         op=torch.distributed.ReduceOp.SUM) if torch.distributed.is_initialized() else None
    last_group_size = len(val_loader.dataset) % world_size

    with torch.no_grad():
        sad = []
        mse = []
        grad = []
        conn = []
        avg_frame_rate = 0.0
        # scale = 0.5
        stride = cfg.MODEL.stride
        epoch_result_dir = os.path.join(cfg.TRAIN.result_dir, str(epoch))
        if epoch > 19 and rank == 0:
            if not os.path.exists(epoch_result_dir):
                os.makedirs(epoch_result_dir)
        for i, sample in enumerate(val_loader):
            image, targets = sample['image'], sample['alpha']
            imageinput = image.clone()

            if imageinput.shape[0] > 1 or last_group_size == 0:
                all_reduce = all_reduce
            else:
                all_reduce = partial(torch.distributed.all_reduce, op=torch.distributed.ReduceOp.SUM,
                                     group=torch.distributed.new_group(range(last_group_size)))

            h, w = image.size()[2:]
            image = image.squeeze().numpy().transpose(1, 2, 0)
            imsize = np.asarray(image.shape[:2], dtype=np.float)
            newsize = np.ceil(imsize / stride) * stride
            padh = int(newsize[0]) - h
            padw = int(newsize[1]) - w
            image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode="reflect")
            inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))

            # inference
            outputs = net(inputs.cuda())

            outputs = outputs.squeeze().cpu().numpy()

            alpha = outputs[:h, :w]
            alpha = np.clip(alpha, 0, 1) * 255.
            trimap = targets[:, 1, :, :].squeeze().numpy()
            mask = np.equal(trimap, 128).astype(np.float32)
            alpha = (1 - mask) * trimap + mask * alpha
            gt_alpha = targets[:, 0, :, :].squeeze().numpy() * 255.

            if epoch > 19:
                _, image_name = os.path.split(image_list[rank + i * world_size][0])
                Image.fromarray(alpha.astype(np.uint8)).save(
                    os.path.join(epoch_result_dir, image_name)
                )

            # compute loss
            SAD = torch.Tensor([compute_sad_loss(alpha, gt_alpha, mask)]).cuda()
            GRAD = torch.Tensor([compute_gradient_loss(alpha, gt_alpha, mask)]).cuda()
            if cfg.VAL.test_all_metrics:
                MSE = torch.Tensor([compute_mse_loss(alpha, gt_alpha, mask)]).cuda()
                CONN = torch.Tensor([compute_connectivity_loss(alpha, gt_alpha, mask)]).cuda()

            count = imageinput.new_tensor([imageinput.shape[0]], dtype=torch.long).cuda()
            if all_reduce:
                all_reduce(count)
            for meter, val in (sad, SAD), (grad, GRAD):
                if all_reduce:
                    all_reduce(val)
                val /= count.item()
                meter.append(val.item())
            if cfg.VAL.test_all_metrics:
                for meter, val in (mse, MSE), (conn, CONN):
                    if all_reduce:
                        all_reduce(val)
                    val /= count.item()
                    meter.append(val.item())

            running_frame_rate = 1 * float(1 / (time() - end))  # batch_size = 1
            end = time()
            avg_frame_rate = (avg_frame_rate * i + running_frame_rate) / (i + 1)
            if rank == 0:
                if i % cfg.TRAIN.print_every == cfg.TRAIN.print_every - 1:
                    print(
                        'epoch: {0}, test: {1}/{2}, sad: {3:.2f}, SAD: {4:.2f}, MSE: {5:.4f}, '
                        'Grad: {6:.2f}, Conn: {7:.2f}, frame: {8:.2f}Hz/{9:.2f}Hz'
                            .format(epoch, i + 1, len(val_loader), sad[-1], np.mean(sad), np.mean(mse),
                                    np.mean(grad), np.mean(conn), running_frame_rate, avg_frame_rate)
                    )
    if imageinput.shape[0] == 1 and rank > last_group_size > 0:
        torch.distributed.new_group(range(last_group_size))
    # write to files
    if rank == 0:
        with open(os.path.join(cfg.TRAIN.result_dir, cfg.EXP + '.txt'), 'a') as f:
            print(
                'epoch: {0}, test: {1}/{2}, SAD: {3:.2f}, MSE: {4:.4f}, Grad: {5:.2f}, Conn: {6:.2f}'
                    .format(epoch, i + 1, len(val_loader), np.mean(sad), np.mean(mse), np.mean(grad), np.mean(conn)),
                file=f
            )
    # save stats
    net.val_loss['epoch_loss'].append(np.mean(sad))
    net.measure['sad'].append(np.mean(sad))
    net.measure['mse'].append(np.mean(mse))
    net.measure['grad'].append(np.mean(grad))
    net.measure['conn'].append(np.mean(conn))


def save_checkpoint(state, snapshot_dir, filename='.pth'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))


def weighted_loss(pd, gt, wl=0.5, epsilon=1e-6):
    bs, _, h, w = pd.size()
    mask = gt[:, 1, :, :].view(bs, 1, h, w)
    alpha_gt = gt[:, 0, :, :].view(bs, 1, h, w)
    diff_alpha = (pd - alpha_gt) * mask
    loss_alpha = torch.sqrt(diff_alpha * diff_alpha + epsilon ** 2)
    loss_alpha = loss_alpha.sum(dim=2).sum(dim=2) / (mask.sum(dim=2).sum(dim=2) + 1)
    loss_alpha = loss_alpha.sum() / bs

    fg = gt[:, 2:5, :, :]
    bg = gt[:, 5:8, :, :]
    c_p = pd * fg + (1 - pd) * bg
    c_g = gt[:, 8:11, :, :]
    diff_color = (c_p - c_g) * mask
    loss_composition = torch.sqrt(diff_color * diff_color + epsilon ** 2)
    loss_composition = loss_composition.sum(dim=2).sum(dim=2) / (mask.sum(dim=2).sum(dim=2) + 1)
    loss_composition = loss_composition.sum() / bs

    return wl * loss_alpha + (1 - wl) * loss_composition



if __name__ == "__main__":
    main()











