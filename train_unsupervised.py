from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np

import time
from NetworkInNetwork import Regressor
from dataset import CIFAR10
import PIL

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/home/lhzhang/cifar10', help='path to dataset')
parser.add_argument('--prefix', default='S4', type=str, help='Mark different models')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--optimizer', default='', help="path to optimizer (to continue training)")
parser.add_argument('--outf', default='./AET/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1814, help='manual seed')
parser.add_argument('--rot', type=float, default=180, help='range of angle for rotation, default:[-180, 180]')
parser.add_argument('--shear', type=float, default=30, help='range of angle for shearing, default: [-30, 30]')
parser.add_argument('--translate', type=float, default=0.2,
                    help='range of ratio of translation to the height/width of the image')
parser.add_argument('--shrink', type=float, default=0.7, help='the lower bound of scaling')
parser.add_argument('--enlarge', type=float, default=1.3, help='the higher bound of scaling')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = CIFAR10(root=opt.dataroot, degrees=opt.rot, translate=(opt.translate, opt.translate),
                        scale=(opt.shrink, opt.enlarge), shear=opt.shear, fillcolor=(128, 128, 128), download=True,
                        resample=PIL.Image.BILINEAR,
                        transform_pre=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                        ]),
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))

assert train_dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=int(opt.workers))

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "5, 7, 3, 2, 1, 0"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ngpu = int(opt.ngpu)

net = Regressor(_num_stages=4, _use_avg_on_conv3=False).to(device)

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=range(ngpu))

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

print(net)

criterion = nn.MSELoss()

# setup optimizer
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(opt.optimizer))

best_err = 10000
best_epoch = 0
for epoch in range(opt.niter):
    # adjust learning rate
    if epoch >= 240 and epoch < 480:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr * 0.2
    elif epoch >= 480 and epoch < 640:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr * 0.04
    elif epoch >= 640 and epoch < 800:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr * 0.008
    elif epoch >= 800 and epoch < 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr * 0.0016
    elif epoch >= 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr * 0.0016 - 3e-7 * (epoch - 999)

    err_epoch = []
    start_time = time.time()
    for i, data in enumerate(train_dataloader, 0):
        net.zero_grad()
        img1 = data[0].to(device)  # original images
        img2 = data[1].to(device)  # tranformed images
        matrix = data[2].to(device)  # transformation matrix
        batch_size = img1.size(0)
        f1, f2, output = net(img1, img2)

        err = criterion(output.to(device), matrix)
        err_epoch.append(err.item())

        err.backward()
        optimizer.step()

        #print('[%d/%d][%d/%d] Loss: %.4f'
        #      % (epoch, opt.niter, i, len(train_dataloader),
        #         err.item()))

    err_avg = np.mean(err_epoch)
    end_time = time.time()
    epoch_time = (end_time-start_time)/60

    remain_time = (opt.niter - epoch) * epoch_time

    print('[%d/%d] Remain time: %.2fmin | Every Epoch %.2fmin | Epoch Err %.4f | best epoch %d'
          % (epoch, opt.niter, remain_time, epoch_time, err_avg, best_epoch))

    if err_avg < best_err:
        best_err = err_avg
        best_epoch = epoch
        torch.save(net.state_dict(), '%s/%s_net_best.pth' % (opt.outf, opt.prefix))
        torch.save(optimizer.state_dict(), '%s/%s_optimizer_best.pth' % (opt.outf, opt.prefix))
        np.savetxt('%s/best_epoch.txt'%(opt.outf), [int(best_epoch)])

    # do checkpointing
    if epoch % 100 == 99:
        torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch))

