from parser import ParserArgs

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.data import DataLoader
from datasets import *

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch

from models.unet_model import UNet
from models.resnet_1 import *
from visualizer import Visualizer

transform_ = transforms.Compose([transforms.Resize((224, 224), interpolation=2),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Baseline_Model(object):
    def __init__(self):
        self.val_best_acc = 0

        args = ParserArgs().args

        model = resnet101(num_classes=3, channels=1)
        model = nn.DataParallel(model).cuda()
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adadelta(model.parameters(), args.lr,
        #                                  weight_decay=args.weight_decay)

        # Optionally resume from a checkpoint
        if args.resume:
            ckpt_root = os.path.join('checkpoints', args.version)
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        self.vis = Visualizer(server='http://10.10.10.100', env='{}'.format(args.version), port=args.port)

        self.dataset_name = args.dataset
        self.train_loader = get_dataloader(self.dataset_name, args.data_path, batch_size=args.batch_size, mode='train',
                                           transform=transform_)
        self.val_loader =get_dataloader(self.dataset_name, args.data_path, batch_size=args.batch_size, mode='val',
                                        transform=transform_)
        self.test_loader =get_dataloader(self.dataset_name, args.data_path, batch_size=args.batch_size, mode='test',
                                         transform=transform_)
        print_args(args)
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.cls_list = {}

    def train_val_test(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            adjust_lr(self.args.lr, self.optimizer, epoch, 50)
            self.epoch = epoch
            self.train(epoch)
            self.val(self.epoch)
            # self.test(self.test_loader_con, 'contentloss')
            # self.test(self.test_loader_lst, 'lstyle')
            # self.test(self.test_loader_our, 'ours')
            # self.test(self.test_loader_bigan, 'bigan')
            # self.test(self.test_loader_cycle, 'cyclegan')
            # self.test(self.test_loader_demo, 'demo')
            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('Node: {}'.format(self.args.node))
            print('Version: {}\n'.format(self.args.version))
        self.test()

    def train(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        self.model.train()
        for i, (input, target) in enumerate(self.train_loader):
            target = target.cuda(async=True)
            input = input.cuda(async=True)

            # compute output
            output = self.model(input)
            loss = self.criterion(output, target.long())

            # measure accuracy and record loss
            acc = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data[0], input.size(0))
            top1.update(acc[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time

            # iter_num =range(len(loss_static))
            if i % 10 == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                      epoch + 1, i, len(self.train_loader), loss=losses, top1=top1))
            self.vis.plot('train_loss', loss.data)
        self.vis.plot('train_acc', top1.avg)

    def val(self, epoch):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()

        output_all = torch.FloatTensor([]).cuda()
        target_all = torch.FloatTensor([]).cuda()
        self.model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                target = target.cuda(async=True)
                input = input.cuda(async=True)
                # compute output
                output = self.model(input)
                output_all = torch.cat((output_all, output.data))
                target_all = torch.cat((target_all, target.type_as(target_all)))

                acc = accuracy(output.data, target, topk=(1,))
                acc_meter.update(acc[0], input.size()[0])

                loss = self.criterion(output, target.long())
                loss_meter.update(loss.data, input.size()[0])
                print('Val Epoch: [{0}][{1}/{2}]\t'
                      'Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                       epoch + 1, i, len(self.val_loader), loss=loss_meter, top1=acc_meter))
                self.vis.plot('val_loss', loss.data)
            cls_list, acc_list = accuracy_per_class(output_all, target_all, self.dataset_name)
            is_best = acc_meter.avg > self.val_best_acc

            self.val_best_acc = max(acc_meter.avg, self.val_best_acc)
            save_ckpt(version=self.args.version,
                      state={'epoch': self.epoch+1,
                             'state_dict': self.model.state_dict(),
                             'best_acc': self.val_best_acc,
                             'optimizer': self.optimizer.state_dict(),
                             }, is_best=is_best & (epoch >= 10),
                      epoch=self.epoch+1)
            if is_best:
                torch.save(self.model.state_dict(), 'checkpoints/{}/{}_best_model.pkl'
                           .format(self.args.version, self.args.version))
            print('Save ckpt successfully!')
            self.vis.text('best val acc:{}%\n'.format(self.val_best_acc), name='val result')
            self.vis.text('Args: \n{}\n'.format(self.args), name='args information')
            self.vis.plot('val_acc', acc_meter.avg)
            self.vis.plot_legend('val_acc_per_classes', acc_list, cls_list)

    def test(self):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()

        output_all = torch.FloatTensor([]).cuda()
        target_all = torch.FloatTensor([]).cuda()
        self.model.load_state_dict(torch.load('checkpoints/{}/best_model.pkl'.format(self.args.version)))
        self.model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                target = target.cuda(async=True)

                # compute output
                output = self.model(input)
                output_all = torch.cat((output_all, output.data))
                target_all = torch.cat((target_all, target.type_as(target_all)))

                acc = accuracy(output.data, target, topk=(1,))
                acc_meter.update(acc[0], input.size()[0])

                loss = self.criterion(output, target.long())
                loss_meter.update(loss.data, input.size()[0])
                print('Test Loss[{}]:{loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                       i, loss=loss_meter, top1=acc_meter))
                self.vis.plot('test_loss\n', loss.data)
            cls_list, acc_list = accuracy_per_class(output_all, target_all, self.dataset_name)
            self.vis.text('test loss:{}\n  test acc:{}%\n  {} acc:{}%\n  {} acc:{}%\n  {} acc:{}%\n'.
                          format(loss_meter.avg, acc_meter.avg, cls_list[0], acc_list[0], cls_list[1], acc_list[1],
                                 cls_list[2], acc_list[2]), name='val result')


def main():
    # tb(Traceback.colour) function should be removed
    # import tb
    # tb.colour()

    Baseline_Model().train_val_test()


if __name__ == '__main__':
    main()