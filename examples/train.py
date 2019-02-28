import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data

import torchvision

from tensorboardX import SummaryWriter

from ERFNet import ERFNet, BoxERFNet
from ENet import ENet, BoxENet

model_names = ['ENet', 'BoxENet', 'ERFNet', 'BoxERFNet']

parser = argparse.ArgumentParser(
    description='Simple Cityscapes semantic segmentation training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='BoxENet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 12)')

# optimizer
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPT',
                    help='optimizer type (default: Adam)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float, # 0.1
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=None, type=float, metavar='M', # 0.9
                    help='momentum')
parser.add_argument('--nesterov', default=None, type=bool, # True
                    help='use Nesterov momentum?')
parser.add_argument('--weight-decay', '--wd', default=None, type=float, # 1e-4
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--run-name', default=time.ctime(time.time())[4:-8], type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--lr-test', action='store_true',
                    help='do a learning rate test to prepare for superconvergence runs')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_classIoU = 0


def main():
    global args, best_classIoU
    args = parser.parse_args()
    
    random.seed(666)
    torch.manual_seed(666)
  
    from datasets import Cityscapes
    
    train_dataset = Cityscapes(
        args.data, split='train', size=(512, 256), augmented=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    train_loader.iter = iter(train_loader)

    val_dataset = Cityscapes(
        args.data, split='val', size=(512, 256), augmented=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print('Architecture:', args.arch)
    class ModelWithLogSoftmax(globals()[args.arch]):
        def forward(self, x):
            heatmap_raw = super().forward(x)
            return torch.nn.functional.log_softmax(heatmap_raw, dim=1)

    model = ModelWithLogSoftmax(n_classes=19).cuda()

    # model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.NLLLoss(weight=train_dataset.class_weights).cuda()

    optimizer_kwargs = {hyperparam: getattr(args, hyperparam) \
        for hyperparam in ('lr', 'weight_decay', 'momentum', 'nesterov') \
        if getattr(args, hyperparam) is not None}
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), **optimizer_kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_classIoU = checkpoint['best_classIoU']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise "=> no checkpoint found at '{}'".format(args.resume)

    torch.backends.cudnn.benchmark = True

    board_writer = SummaryWriter(os.path.join('runs', args.run_name))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        val_loader.iter = iter(val_loader)
        # train for one epoch
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, board_writer)
        
        train_loader.iter = iter(train_loader)
        # evaluate on validation set
        val_metrics = validate(val_loader, model, criterion)

        # record metrics to tensorboard
        for tra,val,name in zip(train_metrics, val_metrics, ('Class IoU', 'Category IoU', 'Loss')):
            board_writer.add_scalars(name, {'Train': tra, 'Test': val}, epoch+1)
        
        # remember best score and save checkpoint
        classIoU = val_metrics[0]
        is_best = classIoU > best_classIoU
        best_classIoU = max(classIoU, best_classIoU)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_classIoU': best_classIoU,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join('runs', args.run_name, ))


def train(train_loader, model, criterion, optimizer, epoch, board_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    classIoU_meter = AverageMeter()
    categIoU_meter = AverageMeter()

    # switch to train mode
    model.train()

    total_batches = len(train_loader) if not args.lr_test else 9

    end = time.time()
    for i, (input, target) in enumerate(train_loader.iter):
        if i > total_batches: break

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        global_iteration = epoch * total_batches + i
        adjust_learning_rate(optimizer, epoch, i, total_batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        with torch.no_grad():
            if i % 80 == 0:
                # measure accuracy and record loss
                classIoU, categIoU = accuracy(output, target)

                import numpy as np
                image = input[0].cpu().numpy().transpose((1,2,0)).copy()
                image -= image.min()
                image /= image.max()
                image *= 255.
                image = image.astype(np.uint8)
                output_map = output[0].max(0)[1].cpu().numpy()
                import imgaug
                segmap_display = imgaug.SegmentationMapOnImage(output_map, image.shape[:2], 20).draw_on_image(image)
                segmap_display = segmap_display.transpose((2,0,1)).copy()

                board_writer.add_image('Example segmentation', segmap_display, global_iteration)

                loss_meter.update(loss.item(), input.size(0))
                classIoU_meter.update(classIoU, target.numel())
                categIoU_meter.update(categIoU, target.numel())

                board_writer.add_scalar('Learning rate',
                    next(iter(optimizer.param_groups))['lr'], global_iteration)
                board_writer.add_scalars('Time',
                    {'Total batch time': batch_time.val,
                     'Data loading time': data_time.val}, global_iteration)
                board_writer.add_scalar('Online batch loss', loss_meter.val, global_iteration)
                board_writer.add_scalars('Online accuracies',
                    {'Class IoU': classIoU_meter.val,
                     'Category IoU': categIoU_meter.val}, global_iteration)

    return classIoU_meter.avg, categIoU_meter.avg, loss_meter.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    classIoU_meter = AverageMeter()
    categIoU_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader.iter):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            classIoU, categIoU = accuracy(output, target)
            loss_meter.update(loss.item(), input.size(0))
            classIoU_meter.update(classIoU, target.numel())
            categIoU_meter.update(categIoU, target.numel())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return classIoU_meter.avg, categIoU_meter.avg, loss_meter.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    assert filename.endswith('.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + '_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epoch_iteration=None, iters_per_epoch=None):
    #"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))

    # Super-Convergence
    MAX_LR = 3.2292e-3 #4.5e-3
    MIN_LR = MAX_LR / 12
    END_LR = 3e-7
    total_iterations = args.epochs * iters_per_epoch

    stage1_start = round(0.0 * args.epochs * 5/6) * iters_per_epoch
    stage2_start = round(0.5 * args.epochs * 5/6) * iters_per_epoch
    stage3_start = round(1.0 * args.epochs * 5/6) * iters_per_epoch

    global_iteration = epoch*iters_per_epoch + epoch_iteration

    if global_iteration < stage2_start:
        lr = MIN_LR + (MAX_LR-MIN_LR) * (global_iteration-stage1_start) / (stage2_start-stage1_start)
    elif global_iteration < stage3_start:
        lr = MAX_LR - (MAX_LR-MIN_LR) * (global_iteration-stage2_start) / (stage3_start-stage2_start)
    else:
        lr = MIN_LR - (MIN_LR-END_LR) * (global_iteration-stage3_start) / (total_iterations-stage3_start)

    # LR range test
    if args.lr_test:
        lr = epoch * 3e-2 / args.epochs

    lr = args.lr * (0.2 ** (epoch // 100))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    # output: B x Cl+1 x H x W
    # target: B        x H x W
    with torch.no_grad():
        n_classes = output.shape[1] - 1

        output = output.max(1)[1].view(-1)
        target = target.view(-1)
        confusion_matrix = torch.bincount(target*(n_classes+1) + output, minlength=(n_classes+1)**2)
        confusion_matrix = confusion_matrix.view(n_classes+1, n_classes+1)[1:, 1:].cpu()

        class_categories = [
            [0, 1],                   # flat
            [2, 3, 4],                # construction
            [5, 6, 7],                # object
            [8, 9],                   # nature
            [10],                     # sky
            [11, 12],                 # human
            [13, 14, 15, 16, 17, 18], # vehicle
        ]

        classIoU = torch.empty(n_classes, dtype=torch.float32)
        for class_idx in range(n_classes):
            TP = confusion_matrix[class_idx, class_idx].item()
            FN = confusion_matrix[class_idx, :].sum().item() - TP
            FP = confusion_matrix[:, class_idx].sum().item() - TP

            classIoU[class_idx] = TP / max(TP + FP + FN, 1)

        categoryIoU = torch.empty(len(class_categories), dtype=torch.float32)
        for category_idx, category in enumerate(class_categories):
            TP = 0
            for class_idx in category:
                TP += confusion_matrix[class_idx, class_idx].item()

            FN, FP = -TP, -TP
            for class_idx in category:
                FN += confusion_matrix[class_idx, :].sum().item()
                FP += confusion_matrix[:, class_idx].sum().item()

            categoryIoU[category_idx] = TP / max(TP + FP + FN, 1)

        return classIoU.mean(), categoryIoU.mean()


if __name__ == '__main__':
    main()
