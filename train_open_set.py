from parser import ParserArgs
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.data import DataLoader
from datasets import *

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import autograd
from sklearn.metrics import roc_auc_score

from models.resnet_1 import *
from models.pix2pix_model import *
from visualizer import Visualizer

# transform_ = transforms.Compose([transforms.Resize((256, 256), interpolation=2),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_cls = transforms.Compose([transforms.Resize((224, 224), interpolation=2),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Openset_train_Model(object):
    def __init__(self):
        self.val_best_acc = 0
        self.val_best_auc = 0.5

        args = ParserArgs().args

        encoder = Encoder(in_channels=1, out_channels=1)
        generator = Generator(out_channels=1)
        discriminator = Discriminator_openset()
        classifier = resnet101(num_classes=3, channels=1)
        encoder = nn.DataParallel(encoder).cuda()
        generator = nn.DataParallel(generator).cuda()
        discriminator = nn.DataParallel(discriminator).cuda()
        classifier = nn.DataParallel(classifier).cuda()
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_C = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        milestones = [40, 120]
        lr_scheduler_E = torch.optim.lr_scheduler.MultiStepLR(optimizer_E, milestones=milestones, gamma=0.1)
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=milestones, gamma=0.1)
        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=milestones, gamma=0.1)
        # Optionally resume from a checkpoint
        if args.resume:
            ckpt_root = os.path.join('checkpoints', args.version)
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                encoder.load_state_dict(checkpoint['state_dict_E'])
                generator.load_state_dict(checkpoint['state_dict_G'])
                discriminator.load_state_dict(checkpoint['state_dict_D'])
                classifier.load_state_dict(checkpoint['state_dict_C'])
                optimizer_E.load_state_dict(checkpoint['optimizer_E'])
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                optimizer_C.load_state_dict(checkpoint['optimizer_C'])
                self.val_best_acc = checkpoint['val_best_acc']

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        self.vis = Visualizer(server='http://10.10.10.100', env='{}'.format(args.version), port=args.port)

        self.dataset_name = args.dataset
        self.train_loader = get_dataloader(self.dataset_name, args.data_path, batch_size=args.batch_size, mode='train',
                                           transform=transform_cls)
        self.val_loader =get_dataloader(self.dataset_name, args.data_path, batch_size=args.batch_size, mode='val',
                                        transform=transform_cls)
        self.test_loader =get_dataloader(self.dataset_name, args.data_path, batch_size=args.batch_size, mode='test',
                                         transform=transform_cls)
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
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier

        self.optimizer_E = optimizer_E
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_C = optimizer_C

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_GAN = nn.MSELoss()
        self.criterion_pixwise = nn.L1Loss()
        self.lr_scheduler_E = lr_scheduler_E
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D

        self.lambda_pixel = 100
        self.cls_list = {}

    def sample_images(self, batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(self.val_loader))
        real = imgs[0].type(torch.cuda.FloatTensor)
        fake = self.generator(self.encoder(real))
        img_sample = torch.cat((real.data, fake.data), -2)
        os.makedirs(os.path.join('images/', self.args.version), exist_ok=True)
        save_image(img_sample, 'images/%s/%s.png' % (self.args.version, batches_done), nrow=5, normalize=True)

    def train_val_test(self):
        if self.args.mode == 'train_GAN':
            for epoch in range(self.args.epochs):
                # adjust_lr(self.args.lr, self.optimizer_E, self.optimizer_D, epoch, 50)
                # adjust_lr(self.args.lr, self.optimizer_G, self.optimizer_D, epoch, 50)
                # adjust_lr(self.args.lr, self.optimizer_D, self.optimizer_D, epoch, 50)
                self.train_GAN(epoch)
                print('\n', '*' * 10, 'Program Information', '*' * 10)
                print('Node: {}'.format(self.args.node))
                print('Version: {}\n'.format(self.args.version))
        elif self.args.mode == 'generate_unknown':
            self.generate_unknown()
            self.train_loader = get_dataloader(self.dataset_name, self.args.data_path, batch_size=self.args.batch_size,
                                               mode='train_unknown', transform=transform_cls)
            for epoch in range(self.args.epochs):
                adjust_lr(self.args.lr, self.optimizer_C, epoch, 40)
                self.vis.text('Args: \n{}\n'.format(self.args), name='args information')
                self.train(epoch)
                self.val(epoch)
                self.test()
                print('\n', '*' * 10, 'Program Information', '*' * 10)
                print('Node: {}'.format(self.args.node))
                print('Version: {}\n'.format(self.args.version))
        else:
            raise print('unknown operation')

    def train_GAN(self, epoch):
        losses_total = AverageMeter()
        losses_GAN = AverageMeter()
        losses_pixwise = AverageMeter()
        # top1 = AverageMeter()

        patch = (1, 224 // 2 ** 4, 224 // 2 ** 4)
        # switch to train mode
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()
        for i, (input, target) in enumerate(self.train_loader):
            target = target.cuda(async=True)
            real = input.cuda(async=True)
            valid_label = torch.cuda.FloatTensor(np.ones((real.size(0), *patch)))
            fake_label = torch.cuda.FloatTensor(np.zeros((real.size(0), *patch)))

            #############################
            # update  generator
            #############################
            # compute output
            fake = self.generator(self.encoder(real))
            pred_fake = self.discriminator(fake)
            loss_GAN = self.criterion_GAN(pred_fake, valid_label)
            loss_pixwise = self.criterion_pixwise(fake, real)

            loss_EG = loss_GAN + loss_pixwise * self.lambda_pixel

            losses_GAN.update(loss_GAN, real.size(0))
            losses_pixwise.update(loss_pixwise, real.size(0))
            losses_total.update(loss_EG, real.size(0))

            self.optimizer_E.zero_grad()
            self.optimizer_G.zero_grad()
            self.optimizer_G.zero_grad()

            loss_EG.backward()
            self.optimizer_E.step()
            self.optimizer_G.step()
            self.vis.plot_legend('Genertor-train', [loss_EG.data, loss_GAN.data, loss_pixwise.data],
                                 ['EG_loss', 'loss_GAN', 'loss_pixel'])

            #############################
            # update discriminator
            #############################
            pred_real = self.discriminator(real)
            loss_real = self.criterion_GAN(pred_real, valid_label)

            pred_fake = self.discriminator(fake.detach())
            loss_fake = self.criterion_GAN(pred_fake, fake_label)

            loss_D = 0.5 * (loss_real + loss_fake)
            self.vis.plot_legend('Discriminator-train', [100 * loss_D.data, loss_real.data, loss_fake.data],
                                 ['D_loss', 'real_loss', 'fake_loss'])

            self.optimizer_D.zero_grad()
            loss_D.backward()

            self.optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(self.train_loader) + i
            batches_left = self.args.epochs * len(self.train_loader) - batches_done

            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [EG loss: %f, pixel: %f, adv: %f]" %
                             (epoch, self.args.epochs, i, len(self.train_loader),
                              loss_D.item(), loss_EG.item(), loss_pixwise.item(), loss_GAN.item()))

            # print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
            #       (epoch, opt.n_epochs,
            #        i, len(dataloader),
            #        loss_D.item(), loss_G.item(),
            #        loss_pixel.item(), loss_GAN.item(),
            #        time_left))
            # # If at sample interval save image
            if batches_done % 20 == 0:
                self.sample_images(batches_done)
            if i % 10 == 0:
                vis_real = real * 0.5 + 0.5
                vis_fake = fake * 0.5 + 0.5
                visualize_image = torch.cat([vis_real[:2].unsqueeze(0), vis_fake[:2].unsqueeze(0)], 0)
                # image_saver(os.path.join('results', result_dir_name, 'prediction'), visualize_image, counter.count)
                self.vis.images(vis_real[:6], 'Training:inputs ')
                self.vis.images(vis_fake[:6], 'Training:output')

        self.lr_scheduler_E.step()
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()
        if epoch % 10 == 0:
            save_ckpt(version=self.args.version, project='train-GAN-{}'.format(epoch),
                      state={'epoch': epoch + 1,
                             'state_dict_E': self.encoder.state_dict(),
                             'state_dict_G': self.generator.state_dict(),
                             'state_dict_D': self.discriminator.state_dict(),
                             'state_dict_C': self.classifier.state_dict(),
                             'optimizer_E': self.optimizer_E.state_dict(),
                             'optimizer_G': self.optimizer_G.state_dict(),
                             'optimizer_D': self.optimizer_D.state_dict(),
                             }, epoch=epoch+1)

    def generate_unknown(self):
        checkpoint = torch.load('checkpoints/{}/v01_01_Baseline_cell_ckpt.pth.tar'.format(self.args.version))
        self.classifier.load_state_dict(checkpoint['state_dict'])
        print('load classifier successful')
        self.encoder.eval()
        self.generator.eval()
        max_iter = 100
        image_name = 1
        for i, (input, target) in enumerate(self.train_loader):
            target = target.cuda(async=True)
            input = input.cuda(async=True)

            latent = self.encoder(input)
            latent_unknown = latent.clone()
            for j in range(max_iter):
                unknown_image = self.generator(latent)
                cls_loss = F.nll_loss(F.log_softmax(self.classifier(unknown_image), dim=1), target.long())
                latent_loss = torch.mean((latent.mean(dim=-1).mean(dim=-1) - latent_unknown.mean(dim=-1).mean(dim=-1)) ** 2)
                total_loss = latent_loss + cls_loss
                pdb.set_trace()
                d_latent_unknown = autograd.grad(total_loss, latent_unknown)
                print(d_latent_unknown[0])
                latent_unknown = latent_unknown - d_latent_unknown[0] * 0.1
                print('Iteration:{}\tcls_loss:{}\tlatent_loss:{}\ttotal_loss:{}'
                      .format(j, cls_loss, latent_loss, total_loss))
                self.vis.plot_legend('cls-latent-train', [cls_loss.data, latent_loss.data, total_loss.data],
                                     ['cls_loss', 'latent_loss', 'total_loss'])

                self.vis.images(unknown_image[:6], 'generate fake unknown')

            for k in range(unknown_image.size(0)):
                save_image(unknown_image[k], '{}/train/unknown/{}.png'.format(self.args.data_path, image_name), nrow=5,
                           normalize=True)
                image_name = image_name + 1

    def train(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        for i, (input, target) in enumerate(self.train_loader):
            target = target.cuda(async=True)
            input = input.cuda(async=True)

            output = self.classifier(input)
            loss = self.criterion_cls(output, target.long())

            self.classifier.zero_grad()
            loss.backward()
            self.optimizer_C.step()

            acc = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data[0], input.size(0))
            top1.update(acc[0], input.size(0))

            if i % 10 == 0:
                sys.stdout.write('Train Epoch: [{0}][{1}/{2}]\t'
                                 'Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                                 'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                                  epoch + 1, i, len(self.train_loader), loss=losses, top1=top1))
            self.vis.plot('train_loss', loss.data)
        self.vis.plot('train_acc', top1.avg)

    def val(self, epoch):
        acc_meter = AverageMeter()
        auc_meter = AverageMeter()
        loss_meter = AverageMeter()

        self.classifier.load_state_dict(torch.load('checkpoints/{}/{}_best_model.pkl'
                                                   .format(self.args.version, self.args.version)))

        output_all = torch.FloatTensor([]).cuda()
        target_all = torch.FloatTensor([]).cuda()
        self.classifier.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                target = target.cuda(async=True)

                # compute output
                output = self.classifier(input)
                output_all = torch.cat((output_all, output.data))
                target_all = torch.cat((target_all, target.type_as(target_all)))

                acc = accuracy(output.data, target, topk=(1,))
                acc_meter.update(acc[0], input.size()[0])

                loss = self.criterion_cls(output, target.long())
                loss_meter.update(loss.data, input.size()[0])
                print('Val Epoch: [{0}][{1}/{2}]\t'
                      'Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                       epoch + 1, i, len(self.val_loader), loss=loss_meter, top1=acc_meter))
                self.vis.plot('val_loss', loss.data)
            cls_list, acc_list = accuracy_per_class(output_all, target_all, self.dataset_name)
            prob = nn.functional.softmax(output_all)
            print('prob size:{}'.format(prob.size()))
            auc = roc_auc_score(np.array(target_all == 2), np.array(prob.data[:, 2]))

            is_best_acc = acc_meter.avg > self.val_best_acc
            is_best_auc = auc_meter.avg > self.val_best_auc
            self.val_best_acc = max(acc_meter.avg, self.val_best_acc)
            self.val_best_auc = max(auc_meter.avg, self.val_best_auc)
            save_ckpt(version=self.args.version, project='classify-unknow_acc',
                      state={'epoch': epoch+1,
                             'classifier': self.classifier.state_dict(),
                             'val_best_acc': self.val_best_acc,
                             'optimizer_C': self.optimizer_C.state_dict(),
                             }, is_best=is_best_acc & epoch > 2,
                      epoch=epoch+1)
            if is_best_acc:
                torch.save(self.classifier.state_dict(), 'checkpoints/{}/{}_best_acc_model.pkl'
                           .format(self.args.version, self.args.version))
            if is_best_auc:
                torch.save(self.classifier.state_dict(), 'checkpoints/{}/{}_best_auc_model.pkl'
                           .format(self.args.version, self.args.version))
            print('Save ckpt successfully!')
            self.vis.text('best val acc:{}%             best val auc:{}'.format(self.val_best_acc, auc),
                          name='val result')
            # self.vis.plot_many(dict(iou_val=iou.mean(), acc_val=acc_meter.average()))
            self.vis.plot_legend('val_acc_auc', [acc_meter.avg, auc], ['acc, auc'])
            self.vis.plot_legend('val_acc_per_classes', acc_list, cls_list)

    def test(self):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()

        output_all = torch.FloatTensor([]).cuda()
        target_all = torch.FloatTensor([]).cuda()
        self.classifier.load_state_dict(torch.load('checkpoints/{}/{}_best_model_acc.pkl'
                                                   .format(self.args.version, self.args.version)))
        best_model_acc = self.classifier
        self.classifier.load_state_dict(torch.load('checkpoints/{}/{}_best_model_auc.pkl'
                                                   .format(self.args.version, self.args.version)))
        best_model_auc = self.classifier
        self.classifier.eval()
        best_model_acc.eval()
        best_model_auc.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                target = target.cuda(async=True)

                # compute output
                output = best_model_acc(input)
                output_all = torch.cat((output_all, output.data))
                target_all = torch.cat((target_all, target.type_as(target_all)))

                acc = accuracy(output.data, target, topk=(1,))
                acc_meter.update(acc[0], input.size()[0])

                loss = self.criterion_cls(output, target.long())
                loss_meter.update(loss.data, input.size()[0])
                print('best acc model-----Test Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                       loss=loss_meter, top1=acc_meter))
                self.vis.plot('test_loss_best_acc_model', loss.data)

            prob = nn.functional.softmax(output_all)
            print('prob size:{}'.format(prob.size()))
            auc = roc_auc_score(np.array(target_all == 2), np.array(prob.data[:, 2]))

            cls_list, acc_list = accuracy_per_class(output_all, target_all, self.dataset_name)
            self.vis.text('test loss:{}         test acc:{}%        {} acc:{}%      {} acc:{}%      {} acc:{}%      '
                          'test auc:{}'.format(loss_meter.avg, acc_meter.avg, cls_list[0], acc_list[0], cls_list[1],
                                               acc_list[1], cls_list[2], acc_list[2], auc),
                                               name='test result acc best model')

            for i, (input, target) in enumerate(self.test_loader):
                target = target.cuda(async=True)

                # compute output
                output = best_model_auc(input)
                output_all = torch.cat((output_all, output.data))
                target_all = torch.cat((target_all, target.type_as(target_all)))

                acc = accuracy(output.data, target, topk=(1,))
                acc_meter.update(acc[0], input.size()[0])

                loss = self.criterion_cls(output, target.long())
                loss_meter.update(loss.data, input.size()[0])
                print('best auc model-----Test Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc:{top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                       loss=loss_meter, top1=acc_meter))
                self.vis.plot('best_auc_model_test_loss', loss.data)

            prob = nn.functional.softmax(output_all)
            print('prob size:{}'.format(prob.size()))
            auc = roc_auc_score(np.array(target_all == 2), np.array(prob.data[:, 2]))

            cls_list, acc_list = accuracy_per_class(output_all, target_all, self.dataset_name)
            self.vis.text('test loss:{}         test acc:{}%        {} acc:{}%      {} acc:{}%      {} acc:{}%      '
                          'test auc:{}'.format(loss_meter.avg, acc_meter.avg, cls_list[0], acc_list[0], cls_list[1],
                                               acc_list[1], cls_list[2], acc_list[2], auc),
                                               name='test result auc best model')


def main():
    # tb(Traceback.colour) function should be removed
    # import tb
    # tb.colour()

    Openset_train_Model().train_val_test()


if __name__ == '__main__':
    main()