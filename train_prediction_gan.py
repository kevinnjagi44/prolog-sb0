import argparse
import os
import pdb
import logging

from models.Discriminator import *
from datasets import *
from utils import *

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from math import log10
from sklearn.metrics import roc_auc_score
from PIL import Image

from Loss import GradLoss
from get_models import *
from visualizer import Visualizer


parser = argparse.ArgumentParser()
parser.add_argument('--train_dataroot', type=str, default='/root/datasets/OCT_cheng', help='data path')
parser.add_argument('--model', type=str, default='unet', help='model')
parser.add_argument('--epoch', type=int, default=0, help='start epoch')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epoches of training')
parser.add_argument('--batchsize', type=int, default=1, help='size of batches')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--decay_epoch', type=int, default=15, help='decay lr every regular epoch')
parser.add_argument('--resize', type=int, default=256, help='input data resize')
parser.add_argument('--input_nc', type=int, default=1, help='channel of input size')
parser.add_argument('--output_nc', type=int, default=1, help='channel of output size')
parser.add_argument('--input_ni', type=int, default=4, help='number of input images')
parser.add_argument('--num_workers', type=int, default=4, help='numbers of cpus')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold of positive and negative')
parser.add_argument('--visdom_env', type=str, default='main', help='environment of visdom')
parser.add_argument('--run_id', type=str, default='0001', help='directory number you want to reload')
opt = parser.parse_args()
result_dir_name = opt.run_id + '-' + opt.model


# ############   make results directory   #############
results = os.listdir('results')

vis = Visualizer(server='http://10.10.10.100', env=opt.visdom_env, port=31671, use_incoming_socket=True)

# ############   Network   ############## #
Generator = get_model('unet', opt.input_nc * opt.input_ni, opt.output_nc)
Discriminator = Discriminator_cycle(opt.input_nc)

if result_dir_name in results:
    Generator.load_state_dict(torch.load(os.path.join(opt.run_id, 'Generator-{}.pkl'.format(opt.epoch))))
    Discriminator.load_state_dict(torch.load(os.path.join(opt.run_id, 'Discriminator-{}.pkl'.format(opt.epoch))))
else:
    os.mkdir(os.path.join('results', result_dir_name))
    os.mkdir(os.path.join('results', result_dir_name, 'prediction'))

# coding=utf-8
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
logfile = './results/{}/logger.txt'.format(result_dir_name)
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG) # 输出到file的log等级的开关
# 第三步，再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING) # 输出到console的log等级的开关
# 第四步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 第五步，将logger添加到handler里面
logger.addHandler(fh)
logger.addHandler(ch)

logging.info(opt)

Generator.cuda()
Discriminator.cuda()

# Generator.apply(weights_init_normal)
# Discriminator.apply(weights_init_normal)

# Loss
criterion_int = torch.nn.MSELoss()
criterion_grd = GradLoss()
criterion_GAN = torch.nn.MSELoss()

# Optimizers & lr schedulers
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                   opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                   opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
target_real = Variable(Tensor(opt.batchsize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchsize).fill_(0.0), requires_grad=False)

# Dataloader
transform = transforms.Compose([transforms.Resize((opt.resize, opt.resize), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataroot1 = '/root/datasets/OCT_BOE'
dataroot2 = '/root/datasets/OCT_challenge_15/ai_challenger_fl2018_trainingset'
train_dataloader = DataLoader(oct_cheng_dataset(opt.train_dataroot, transform=transform, mode='train', source_num=opt.input_ni),
                              batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers)
val_dataloader = DataLoader(oct_boe_and_challen_dataset(dataroot1, dataroot2, transform=transform, mode='val', source_num=opt.input_ni),
                            batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)
test_dataloader = DataLoader(oct_boe_and_challen_dataset(dataroot1, dataroot2, transform=transform, mode='test', source_num=opt.input_ni),
                             batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

threshold = opt.threshold

counter = Counter()
# #############   Train   ############## #
for epoch in range(opt.epoch, opt.n_epochs):
    Generator.train_GAN()
    Discriminator.train()
    for i, (batch, real_future) in enumerate(train_dataloader):
        input = batch[0].unsqueeze(0).cuda()
        target = real_future[0].unsqueeze(0).cuda()

        # ################   optimaize G  ################## #
        optimizer_G.zero_grad()

        prediction = Generator(input)
        print(prediction)
        # intensity loss and gradient loss
        loss_intensity = criterion_int(prediction, target)
        loss_gradient = criterion_grd(prediction, target)
        G_total_loss = loss_intensity

        G_total_loss.backward()

        optimizer_G.step()

        vis.plot_legend('Genertor-train', [G_total_loss.data, loss_intensity.data, loss_gradient.data],
                        ['G_loss', 'inte_loss', 'grad_loss'])

        print('Epoch[{}/{}] Iter[{}/{}]\tLoss G:{:.4f}\tloss_intensity:{:.4f}\tloss_gradient:{:.4f}'
              .format(opt.n_epochs, epoch + 1, len(train_dataloader), i, G_total_loss, loss_intensity, loss_gradient))
        # ################   optimaize D  ################## #
        if i % 3 == 0:
            optimizer_D.zero_grad()

            # D Loss (fake and real)
            decision_real = Discriminator(target)[0]

            loss_real = criterion_GAN(decision_real, target_real)

            decision_fake = Discriminator(prediction.detach())[0]
            loss_fake = criterion_GAN(decision_fake, target_fake)

            D_total_loss = (loss_fake + loss_real) * 0.5
            D_total_loss.backward()

            optimizer_D.step()
            vis.plot_legend('Discriminator-train', [D_total_loss.data, loss_real.data, loss_fake.data],
                            ['D_loss', 'real_loss', 'fake_loss'])

            print('Epoch[{}/{}] Iter[{}/{}]\tLoss D:{:.4f}\tloss_real:{:.4f}\tloss_fake:{:.4f}'
                  .format(opt.n_epochs, epoch + 1, len(train_dataloader), i, D_total_loss, loss_real, loss_fake))
        ##########################################################
        if i % 64 == 0:
            visualize_image = torch.cat([input.transpose(0, 1), target, prediction], 0) * 0.5 + 0.5
            # image_saver(os.path.join('results', result_dir_name, 'prediction'), visualize_image, counter.count)
            vis.images(visualize_image, 'Training: {} inputs images-ground truth-output'.format(opt.input_ni),
                       nrow=int((opt.input_ni + 2) / 2))

            # counter.step()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

# #############   Validation   ############# #
    best_auc = 0.5

    loss_statistic = torch.FloatTensor([]).cuda()
    label_statistic = torch.FloatTensor([]).cuda()
    psnr_statistic = []
    Generator.eval()
    Discriminator.eval()
    for i, (batch, real_future, label) in enumerate(val_dataloader):
        input = batch[0].unsqueeze(0).cuda()
        target = real_future[0].unsqueeze(0).cuda()

        prediction = Generator(input)

        # intensity loss and gradient loss
        loss_intensity = criterion_int(prediction, target)
        loss_gradient = criterion_grd(prediction, target)

        G_total_loss_val = loss_intensity

        psnr = 10 * log10(1 / loss_intensity.item())

        psnr_statistic.append(psnr)
        if i == 0:
            loss_statistic = torch.Tensor(np.array([G_total_loss_val.data]))
            label_statistic = label
        else:

            loss_statistic = torch.cat((loss_statistic, torch.Tensor(np.array([G_total_loss_val.data]))), 0)
            label_statistic = torch.cat((label_statistic, label))

        # print('Epoch[{}/{}]\tIter[{}/{}]\tLoss G:{:.4f}\tloss_intensity:{:.4f}\tloss_gradient:{:.4}'
        #       .format(opt.n_epochs, epoch + 1, len(val_dataloader), i, G_total_loss_val, loss_intensity, loss_gradient))

        # ###########     Validation visualize     ############# #
        vis.plot_legend('Genertor-val', [G_total_loss_val.data, loss_intensity.data, loss_gradient.data],
                        ['G_loss', 'inte_loss', 'grad_loss'])

        if i % 64 == 0:
            visualize_image = torch.cat([input.transpose(0, 1), target, prediction], 0) * 0.5 + 0.5
            image_saver(os.path.join('results', result_dir_name, 'prediction'), visualize_image, counter.count)
            vis.images(visualize_image, 'Validation: {} inputs images-ground truth-output'.format(opt.input_ni),
                       nrow=int((opt.input_ni + 2) / 2))

            counter.step()
    # ###########    Compute auc and acc sen spe by psnr    ############# #
    print(psnr_statistic)
    St = (np.array(psnr_statistic) - min(psnr_statistic)) / (max(psnr_statistic) - min(psnr_statistic))
    detection_result = torch.FloatTensor(1 * (St > threshold))
    acc = torch.sum(detection_result == label_statistic.type_as(detection_result)).float() / len(detection_result)

    TP = torch.sum((detection_result == label_statistic.type_as(detection_result)) & (detection_result == 1)).float()
    TN = torch.sum((detection_result == label_statistic.type_as(detection_result)) & (detection_result == 0)).float()
    FP = torch.sum((detection_result != label_statistic.type_as(detection_result)) & (detection_result == 1)).float()
    FN = torch.sum((detection_result != label_statistic.type_as(detection_result)) & (detection_result == 0)).float()

    auc = roc_auc_score(np.array(label_statistic), np.array(St))
    sen = TP / (TP + FN)
    spe = TN / (FP + TN)
    # ###########    delta_s    ############# #
    anomalies_aver = torch.sum(St * label_statistic) / torch.sum(label_statistic.double())
    normal_aver = torch.sum(St * (1 - label_statistic)) / (len(St) - torch.sum(label_statistic.double()))
    delta_s = anomalies_aver - normal_aver

    # ###########    Compute acc    ############## #
    val_print = 'Validation[{}/{}]\tauc:{:.5f}\tdelta_s:{:.5f}\tacc:{:.5f}'\
                .format(opt.n_epochs, epoch + 1, auc, delta_s, acc)
    logging.info(val_print)

    # ###########    Visualize   ########### #
    vis.plot_legend('Validation results', [auc, acc, sen, spe], ['auc', 'acc', 'sen', 'spe'])

    # Save model by some epoches
    if epoch % 5 == 0:
        torch.save(Generator.state_dict(), os.path.join('results', result_dir_name, 'Generator-{}.pkl'.format(epoch + 1)))
        torch.save(Discriminator.state_dict(), os.path.join('results', result_dir_name, 'Discriminator-{}.pkl'.format(epoch + 1)))

    # Save best model
    if auc > best_auc:
        best_auc = auc
        torch.save(Generator.state_dict(), os.path.join('results', result_dir_name, 'Best_Generator.pkl'))

# ###########    Test   ########### #
Generator.load_state_dict(torch.load(os.path.join('results', result_dir_name, 'Best_Generator.pkl')))
best_model = Generator

best_auc = 0.5

loss_statistic = torch.FloatTensor([]).cuda()
label_statistic = torch.FloatTensor([]).cuda()
psnr_statistic = []

for i, (batch, real_future, label) in enumerate(test_dataloader):
    input = batch[0].unsqueeze(0).cuda()
    target = real_future[0].unsqueeze(0).cuda()

    test_prediction = best_model(input)

    # intensity loss and gradient loss
    loss_intensity = criterion_int(test_prediction, target)
    loss_gradient = criterion_grd(test_prediction, target)

    G_total_loss_test = loss_intensity

    print('Testing best model: Loss G:{:.5f}\tloss_intensity:{:.5f}\tloss_gradient:{:.5}'
          .format(G_total_loss_test, loss_intensity, loss_gradient))

    psnr = 10 * log10(1 / loss_intensity.item())

    psnr_statistic.append(psnr)
    if i == 0:
        loss_statistic = torch.Tensor(np.array([G_total_loss_test.data]))
        label_statistic = label
    else:
        loss_statistic = torch.cat((loss_statistic, torch.Tensor(np.array([G_total_loss_test.data]))), 0)
        label_statistic = torch.cat((label_statistic, label))

# ###########    Compute auc and acc by psnr    ############# #
St = (np.array(psnr_statistic) - min(psnr_statistic)) / (max(psnr_statistic) - min(psnr_statistic))
test_prediction = torch.FloatTensor(1 * (St > threshold))
acc = torch.sum(test_prediction == label_statistic.type_as(test_prediction)).float() / len(detection_result)

TP = torch.sum((test_prediction == label_statistic.type_as(test_prediction)) & (test_prediction == 1)).float()
TN = torch.sum((test_prediction == label_statistic.type_as(test_prediction)) & (test_prediction == 0)).float()
FP = torch.sum((test_prediction != label_statistic.type_as(test_prediction)) & (test_prediction == 1)).float()
FN = torch.sum((test_prediction != label_statistic.type_as(test_prediction)) & (test_prediction == 0)).float()

auc = roc_auc_score(np.array(label_statistic), np.array(St))
sen = TP / (TP + FN)
spe = TN / (FP + TN)
# ###########    delta_s    ############# #
anomalies_aver = torch.sum(St * label_statistic) / torch.sum(label_statistic.double())
normal_aver = torch.sum(St * (1 - label_statistic)) / (len(St) - torch.sum(label_statistic.double()))
delta_s = anomalies_aver - normal_aver

test_print = 'Testing :auc:{:.5f}\tdelta_s:{:.5f}\tacc:{:.5f}'.format(auc, delta_s, acc)
print(test_print)
logging.INFO(test_print)
