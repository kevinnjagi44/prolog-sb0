from parser import ParserArgs

import datetime
import sys

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
from visualizer import Visualizer

transforms_ = [ transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


class Baseline_Model(object):
    def __init__(self):
        self.val_best_iou = 0

        args = ParserArgs().args

        model = models.resnet101(n_classes=3)
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
            ckpt_root = os.path.join('saved_models', args.project)
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                self.val_best_iou = checkpoint['best_iou']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        self.vis = Visualizer(server='http://10.10.10.100', env='{}'.format(args.version), port=args.port)

        self.train_loader = DataLoader(oct_cheng_dataset(args.cheng_data, transforms_=transforms_),
                                       batch_size=args.batch_size, shuffle=True)
        self.val_loader =DataLoader(oct_edema_dataset(args.edema_data, transforms_=transforms_),
                                    batch_size=args.batch_size, shuffle=True)
        # self.test_loader_wo = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size).data_load()
        # self.test_loader_with = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                 transfer=True).data_load()


        # self.test_loader_con = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                      dataset='contentloss').data_load()
        # self.test_loader_lst = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                     dataset='lstyle').data_load()
        # self.test_loader_our = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                     dataset='ours').data_load()
        # self.test_loader_bigan = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                     dataset='bigan').data_load()
        # self.test_loader_cycle = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                     dataset='cyclegan').data_load()
        # self.test_loader_demo = volumeLoader(volume_root=args.cheng_data, batch=args.batch_size,
        #                                      dataset='demo').data_load()

        print_args(args)
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train_val_test(self):
        for epoch in range(self.args.epochs):
            adjust_lr(self.args.lr, self.optimizer, epoch, 30)
            self.epoch = epoch
            self.train()
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

    def train(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        self.model.train()

        for i, (input, target) in enumerate(self.train_loader):
            target = target.cuda(async=True)
            input = input.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time

            # iter_num =range(len(loss_static))
            if i % 10 == 0:
                logging.info('Training\tTotal time:{hours}\t'
                             'Epoch: [{0}][{1}/{2}]\t'
                             'Time:{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                             'Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                             'Prec:{top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch + 1, i, len(train_loader), hours=hours, minutes=minutes,
                    batch_time=batch_time, loss=losses,
                    top1=top1))

        return losses.avg

    def val(self, epoch):
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for i, (input, mask_true) in enumerate(self.val_loader):
                input = input.cuda(non_blocking=True)
                mask_true = mask_true.cuda(non_blocking=True).float()
                # mask_true = mask_true.cuda(non_blocking=True).float().unsqueeze(dim=1)

                output = self.model(input)

                _, mask_pred = torch.max(output, dim=1)
                # mask_pred= mask_pred.squeeze().float()
                mask_pred = mask_pred.float()

                acc, pix = accuracy(mask_pred, mask_true)
                intersection, union = intersectionAndUnion(mask_pred, mask_true, 4)

                acc_meter.update(acc, pix)
                intersection_meter.update(intersection)
                union_meter.update(union)
                print('[{}] Epoch: [{}][{}/{}]\t, acc: {:.4f}'.format
                      (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       self.epoch, i, self.val_loader.__len__(), acc))
                if i % 20 == 0:
                    images = torch.cat([input[0].unsqueeze(0) * 0.5 + 0.5,
                                        mask_true[0].unsqueeze(0).unsqueeze(0) / 255,
                                       mask_pred[0].unsqueeze(0).unsqueeze(0).float() / 11])
                    self.vis.images(input[:2], name='edema', )
                    self.vis.images(images, name='val', nrow=3)
                if i % 100 == 0:
                    save_image(torch.cat([input[0].unsqueeze(0) * 0.5 + 0.5, mask_true[0].unsqueeze(0).unsqueeze(0) / 255,
                                          mask_pred[0].unsqueeze(0).unsqueeze(0).float() / 11]),
                               'images/%s.png' % (epoch * 250 + i), nrow=5,
                               normalize=True)
            iou = intersection_meter.sum / (union_meter.sum + 1e-10)

            print('[Eval Summary of Validation]:')
            print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
                  .format(iou.mean(), acc_meter.average() * 100))
            for i, _iou in enumerate(iou):
                print('class [{}], IoU: {}'.format(i, _iou))

            is_best = iou.mean() > self.val_best_iou
            self.val_best_iou = max(iou.mean(), self.val_best_iou)
            save_ckpt(version=self.args.version,
                      state={
                'epoch': self.epoch+1,
                'state_dict': self.model.state_dict(),
                'best_iou': self.val_best_iou,
                'optimizer': self.optimizer.state_dict(),
            },
                      is_best=is_best,
                      epoch=self.epoch+1,
                      project='2019_seg')
            print('Save ckpt successfully!')

            # self.vis.plot_many(dict(iou_val=iou.mean(), acc_val=acc_meter.average()))
            self.vis.plot_legend(win='iou', name='val', y=iou.mean())
            self.vis.plot_legend(win='acc', name='val', y=acc_meter.average())

    def test(self, test_loader, dataset):
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        self.model.eval()
        vis_name = dataset
        with torch.no_grad():
            for i, (input, mask_true, img_name) in enumerate(test_loader):
                input = input.cuda(non_blocking=True)
                mask_true = mask_true.cuda(non_blocking=True).float()
                # mask_true = mask_true.cuda(non_blocking=True).float().unsqueeze(dim=1)

                output = self.model(input)
                _, mask_pred = torch.max(output, dim=1)
                # mask_pred= mask_pred.squeeze().float()
                # post-process
                mask_pred = mask_pred.float()
                mask_pred_post = mask_pred

                acc, pix = accuracy(mask_pred_post, mask_true)
                # TODO: confuse: (B, H, W)
                intersection, union = intersectionAndUnion(mask_pred_post, mask_true, 4)

                acc_meter.update(acc, pix)
                intersection_meter.update(intersection)
                union_meter.update(union)

                if i % 5 == 0:
                    images = torch.cat([input[0].unsqueeze(0),
                                        mask_true[0].unsqueeze(0).unsqueeze(0),
                                        # mask_pred[0].unsqueeze(0).unsqueeze(0).float() / 3,
                                        mask_pred_post[0].unsqueeze(0).float()])
                    self.vis.images(images, name=vis_name, nrow=3, img_name=img_name)

            iou = intersection_meter.sum / (union_meter.sum + 1e-10)

            print('[Eval Summary of Testing]:')
            print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
                  .format(iou.mean(), acc_meter.average() * 100))
            for i, _iou in enumerate(iou):
                print('class [{}], IoU: {}'.format(i, _iou))

            self.vis.plot_legend(win='iou', name=dataset, y=iou.mean())
            self.vis.plot_legend(win='acc', name=dataset, y=acc_meter.average())



def main():
    # tb(Traceback.colour) function should be removed
    # import tb
    # tb.colour()

    SegModel().train_val_test()


if __name__ == '__main__':
    main()