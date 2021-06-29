import os.path as osp
import sys

import subprocess
subprocess.call(['pip', 'install', 'cvbase'])
import cvbase as cvb
import torch
from torch.autograd import gradcheck

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from biupdownsample import biupsample_naive, BiupsampleNaive
from biupdownsample import bidownsample_naive, BidownsampleNaive

feat = torch.randn(2, 64, 2, 2, requires_grad=True, device='cuda:0').double()
mask = torch.randn(
    2, 100, 4, 4, requires_grad=True, device='cuda:0').sigmoid().double()


print('Gradcheck for biupsample naive...')
test = gradcheck(BiupsampleNaive(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)
print(test)


feat = torch.randn(
    2, 1024, 100, 100, requires_grad=True, device='cuda:0').float()
mask = torch.randn(
    2, 25, 200, 200, requires_grad=True, device='cuda:0').sigmoid().float()
loop_num = 500

time_naive_forward = 0
time_naive_backward = 0
bar = cvb.ProgressBar(loop_num)
timer = cvb.Timer()
for i in range(loop_num):
    x = biupsample_naive(feat.clone(), mask.clone(), 5, 1, 2)
    torch.cuda.synchronize()
    time_naive_forward += timer.since_last_check()
    x.sum().backward(retain_graph=True)
    torch.cuda.synchronize()
    time_naive_backward += timer.since_last_check()
    bar.update()
forward_speed = (time_naive_forward + 1e-3) * 1e3 / loop_num
backward_speed = (time_naive_backward + 1e-3) * 1e3 / loop_num
print('\nBiupsample naive time forward: '
      f'{forward_speed} ms/iter | time backward: {backward_speed} ms/iter')


# ---------------------------------------------------------------
feat = torch.randn(2, 64, 4, 4, requires_grad=True, device='cuda:0').double()
mask = torch.randn(
    2, 16, 4, 4, requires_grad=True, device='cuda:0').double()



print('Gradcheck for bidownsample naive...')
test = gradcheck(BidownsampleNaive(4, 1, 1), (feat, mask), atol=1e-4, eps=1e-4)
print(test)




feat = torch.randn(
    2, 512, 200, 200, requires_grad=True, device='cuda:0').float()
mask = torch.randn(
    2, 100, 100, 100, requires_grad=True, device='cuda:0').sigmoid().float()
loop_num = 500


time_naive_forward = 0
time_naive_backward = 0
bar = cvb.ProgressBar(loop_num)
timer = cvb.Timer()
for i in range(loop_num):
    x = bidownsample_naive(feat.clone(), mask.clone(), 10, 1, 2)
    torch.cuda.synchronize()
    time_naive_forward += timer.since_last_check()
    x.sum().backward(retain_graph=True)
    torch.cuda.synchronize()
    time_naive_backward += timer.since_last_check()
    bar.update()
forward_speed = (time_naive_forward + 1e-3) * 1e3 / loop_num
backward_speed = (time_naive_backward + 1e-3) * 1e3 / loop_num
print('\nBidownsample naive time forward: '
      f'{forward_speed} ms/iter | time backward: {backward_speed} ms/iter')


