import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops
import torch
import torch.nn as nn
import torch.distributed as dist


# compute the SAD error given a pdiction, a ground truth and a mask
def compute_sad_loss(pd, gt, mask):
    cv.normalize(pd, pd, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(gt, gt, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    error_map = np.abs(pd - gt) / 255.
    loss = np.sum(error_map * mask)
    # the loss is scaled by 1000 due to the large images
    loss = loss / 1000
    return loss


def compute_mse_loss(pd, gt, mask):
    cv.normalize(pd, pd, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(gt, gt, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    error_map = (pd - gt) / 255.
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    return loss


def compute_gradient_loss(pd, gt, mask):
    cv.normalize(pd, pd, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    cv.normalize(gt, gt, 0.0, 255.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    pd = pd / 255.
    gt = gt / 255.
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map * mask) / 10
    return loss


def compute_connectivity_loss(pd, gt, mask, step=0.1):
    cv.normalize(pd, pd, 0, 255, cv.NORM_MINMAX)
    cv.normalize(gt, gt, 0, 255, cv.NORM_MINMAX)
    pd = pd / 255.
    gt = gt / 255.

    h, w = pd.shape

    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]

        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords

        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]

        dist_maps = morphology.distance_transform_edt(omega==0)
        dist_maps = dist_maps / dist_maps.max()
        # lambda_map[flag == 1] = dist_maps.mean()
    l_map[l_map == -1] = 1
    
    # the definition of lambda is fuzzy
    d_pd = pd - l_map
    d_gt = gt - l_map
    # phi_pd = 1 - lambda_map * d_pd * (d_pd >= 0.15).astype(np.float32)
    # phi_gt = 1 - lambda_map * d_gt * (d_gt >= 0.15).astype(np.float32)
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt) * mask) / 1000
    return loss


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    # x1 = x[:, :, 0:3]
    # x2 = x[:, :, 3]
    # new_x1 = cv.resize(x1, dsize=(w,h), interpolation=cv.INTER_CUBIC)
    # new_x2 = cv.resize(x2, dsize=(w,h), interpolation=cv.INTER_NEAREST)
    #
    # new_x2 = np.expand_dims(new_x2, axis=2)
    # new_x = np.concatenate((new_x1, new_x2), axis=2)
    new_x = cv.resize(x, dsize=(w, h), interpolation=cv.INTER_CUBIC)

    return new_x


def image_rescale(x, scale):
    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv.resize(x1, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    new_x2 = cv.resize(x2, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1,new_x2), axis=2)
    return new_x


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return torch.autograd.Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = torch.nn.functional.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return torch.nn.functional.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = torch.nn.functional.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class SingleGPU(nn.Module):
    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.cuda(non_blocking=True))

class TestDistributedSampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = (len(self.dataset) // self.num_replicas) + int(
            (len(self.dataset) % self.num_replicas) < self.rank)

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = torch.arange(0, len(self.dataset))

        # subsample
        indices = indices[self.rank::self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples