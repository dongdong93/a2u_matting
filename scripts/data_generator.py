import cv2
import os
import math
import numbers
import random
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms
import PIL.Image as Image
import lmdb
import pickle

# from   utils import CONFIG
global cfg


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        global cfg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.phase = phase

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype('float32')) / 255.
        alpha = torch.from_numpy(alpha.astype('float32')).unsqueeze(dim=0)

        # normalize image
        mask = np.equal(trimap, 128).astype('float32')
        mask = torch.from_numpy(mask).unsqueeze(dim=0)
        trimap = torch.from_numpy(trimap.astype('float32')).unsqueeze(dim=0)


        if self.phase == "train":
            # convert GBR images to RGB
            fg = torch.from_numpy(sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype('float32')) / 255.
            bg = torch.from_numpy(sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype('float32')) / 255.
            alpha = torch.cat((alpha, mask, fg, bg, image), dim=0)
        else:
            alpha = torch.cat((alpha, trimap), dim=0)
        image = (image - self.mean) / self.std
        image = torch.cat((image, trimap/255.), dim=0)

        sample['image'], sample['alpha'] = image, alpha


        if self.phase == "train":
            del sample['image_name']
            del sample['trimap']
            del sample['fg']
            del sample['bg']
        else:
            del sample['trimap']

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha


        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, name = sample['fg'], sample['alpha'], sample['trimap'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        hbg, wbg = bg.shape[:2]
        ratio = max(h/hbg, w/wbg)
        if ratio > 1:
            bg = cv2.resize(bg, (math.ceil(wbg*ratio), math.ceil(hbg*ratio)), interpolation=cv2.INTER_CUBIC)
            hbg, wbg = bg.shape[:2]
        if h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                fg = cv2.resize(fg, (int(w * ratio), int(h * ratio)),
                                interpolation=cv2.INTER_NEAREST)
                alpha = cv2.resize(alpha, (int(w * ratio), int(h * ratio)),
                                   interpolation=cv2.INTER_NEAREST)
                trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(wbg * ratio), int(hbg * ratio)),
                                interpolation=cv2.INTER_CUBIC)
                h, w = trimap.shape
        unknown_list = list(zip(*np.where(trimap[self.margin:(h - self.margin),
                                          self.margin:(w - self.margin)] == 128)))

        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # self.logger.warning("{} does not have enough unknown area for crop.".format(name))
            left_top = (
            np.random.randint(0, h - self.output_size[0] + 1), np.random.randint(0, w - self.output_size[1] + 1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0], unknown_list[idx][1])

        fg_crop = fg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        alpha_crop = alpha[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        trimap_crop = trimap[left_top[0]:left_top[0] + self.output_size[0],
                      left_top[1]:left_top[1] + self.output_size[1]]

        if len(np.where(trimap == 128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                              "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=cv2.INTER_CUBIC)

        sample['fg'], sample['alpha'], sample['trimap'] = fg_crop, alpha_crop, trimap_crop
        sample['bg'] = bg_crop

        return sample


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        alpha = sample['alpha']
        # Adobe 1K
        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        sample['trimap'] = trimap

        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image

        return sample


class DataGenerator(Dataset):
    def __init__(self, Cfg, phase="train", test_scale="resize", crop_size = 320, augmentation=True):
        global cfg
        cfg = Cfg
        self.phase = phase
        self.crop_size = cfg.TRAIN.crop_size
        self.augmentation = augmentation
        if self.phase == "train":
            self.fg = np.array([os.path.join(cfg.DATASET.data_dir, name) for name in
                       open(cfg.DATASET.train_fg_list).read().splitlines()])
            self.alpha = np.array([os.path.join(cfg.DATASET.data_dir, name) for name in
                          open(cfg.DATASET.train_alpha_list).read().splitlines()])
            if cfg.TRAIN.lmdb:
                self.bgdir = os.path.join(cfg.DATASET.data_dir, 'Training_set/bg.lmdb')
                self.bg_env = lmdb.open(self.bgdir, readonly=True, lock=False, readahead=False, meminit=False)
                self.bg_meta_info = pickle.load(open(os.path.join(self.bgdir, 'meta_info.pkl'), "rb"))
                self.bg_num = len(self.bg_meta_info['keys'])
            else:
                self.bg = np.array([os.path.join(cfg.DATASET.data_dir, name) for name in open(cfg.DATASET.train_bg_list).read().splitlines()])
                self.bg_num = len(self.bg)
            self.fg_num = len(self.fg)
            if cfg.TRAIN.load_data:
                self.fg_load = dict()
                self.alpha_load = dict()
                for idx in range(self.fg_num):
                    fg_name = self.fg[idx]
                    fg_name = fg_name[fg_name.rfind('/') + 1:-4]
                    fg = cv2.imread(self.fg[idx])
                    alpha = cv2.imread(self.alpha[idx], 0).astype(np.float32) / 255.
                    self.fg_load.update({fg_name: fg})
                    self.alpha_load.update({fg_name: alpha})
            if cfg.TRAIN.load_bg:
                self.bg_load = dict()
                for idx in range(self.bg_num):
                    if cfg.TRAIN.lmdb:
                        bg_name = self.bg_meta_info['keys'][idx]
                        bg = self.read_lmdb(self.bg_env, self.bg_meta_info, idx)
                    else:
                        bg_name = self.bg[idx]
                        bg_name = bg_name[bg_name.rfind('/')+1:-4]
                        bg = cv2.imread(self.bg[idx], 1)
                    self.bg_load.update({bg_name: bg})
        else:
            self.test_list = np.array([name.split('\t') for name in open(cfg.DATASET.val_list).read().splitlines()])


        if augmentation:
            train_trans = [
                            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                            RandomHorizontalFlip(),
                            GenTrimap(),
                            RandomCrop((self.crop_size, self.crop_size)),
                            RandomJitter(),
                            Composite(),
                            ToTensor(phase="train") ]
        else:
            train_trans = [ GenTrimap(),
                            RandomCrop((self.crop_size, self.crop_size)),
                            Composite(),
                            ToTensor(phase="train") ]

        if test_scale.lower() == "origin":
            test_trans = [ToTensor()]
        else:
            raise NotImplementedError("test_scale {} not implemented".format(test_scale))

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,20)]

    def read_lmdb(self, env, meta_info, index):
        # read one image
        key = meta_info['keys'][index]
        img = []
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))

            img_flat = np.fromstring(buf, dtype=np.uint8)

            if len(meta_info['resolution']) == 1:
                C, H, W = [int(s) for s in meta_info['resolution'][0].split('_')]
            else:
                C, H, W = [int(s) for s in meta_info['resolution'][index].split('_')]
            img = img_flat.reshape(H, W, C)

        return img

    def __getitem__(self, idx):
        if self.phase == "train":
            if not cfg.TRAIN.load_data:
                fg = cv2.imread(self.fg[idx % self.fg_num])
                alpha = cv2.imread(self.alpha[idx % self.fg_num], 0).astype(np.float32) / 255
            else:
                fg_name = self.fg[idx % self.fg_num]
                fg_name = fg_name[fg_name.rfind('/')+1:-4]

                fg = self.fg_load[fg_name]
                alpha = self.alpha_load[fg_name]
            bg_idx = np.random.randint(0, self.bg_num - 1) if cfg.TRAIN.random_bgidx else idx
            if not cfg.TRAIN.load_bg:
                if cfg.TRAIN.lmdb:
                    bg = self.read_lmdb(self.bg_env, self.bg_meta_info, bg_idx)
                else:
                    bg = cv2.imread(self.bg[bg_idx], 1)
            else:
                if cfg.TRAIN.lmdb:
                    bg_name = self.bg_meta_info['keys'][bg_idx]
                else:
                    bg_name = self.bg[idx % self.bg_num]
                    bg_name = bg_name[bg_name.rfind('/') + 1:-4]
                bg = self.bg_load[bg_name]
            if bg.shape[2]==1:
                bg = np.repeat(bg, 3, axis=2)
            alpha = np.squeeze(alpha)

            if self.augmentation:
                fg, alpha = self._composite_fg(fg, alpha, idx)

            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name}

        else:
            image = cv2.imread(os.path.join(cfg.DATASET.data_dir, self.test_list[idx][0]))
            alpha = cv2.imread(os.path.join(cfg.DATASET.data_dir, self.test_list[idx][1]), 0).astype(np.float32) / 255.
            alpha = alpha[:, :, 0] if alpha.ndim == 3 else alpha
            trimap = cv2.imread(os.path.join(cfg.DATASET.data_dir, self.test_list[idx][2]), 0)
            image_name = os.path.split(self.test_list[idx][0])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap}

        sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            if not cfg.TRAIN.load_data:
                fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
                alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32) / 255.
            else:
                fg2_name = self.fg[idx2 % self.fg_num]
                fg2_name = fg2_name[fg2_name.rfind('/') + 1:-4]
                fg2 = self.fg_load[fg2_name]
                alpha2 = self.alpha_load[fg2_name].astype(np.float32) / 255.
            alpha2 = np.squeeze(alpha2)
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=cv2.INTER_NEAREST)
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=cv2.INTER_NEAREST)

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if np.any(alpha_tmp < 1):
                fg = ((fg.astype(np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None]) * alpha2[:, :, None])) \
                     / (alpha_tmp[:, :, None] + 1e-5)
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (640, 640), interpolation=cv2.INTER_NEAREST)

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return self.bg_num
        else:
            return len(self.test_list)



if __name__ == '__main__':

    from data_generator import DataGenerator
    from torch.utils.data import DataLoader
    from config import cfg

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    cfg.merge_from_file('./config/adobe.yaml')

    dataset = DataGenerator(cfg, phase='train', test_scale='origin', crop_size=cfg.TRAIN.crop_size)
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            sampler=None
        )

    print(len(dataloader))
    for i, data in enumerate(dataloader, 0):
        images, targets = data['image'], data['alpha']
        print(i)
