import os
import glob
import shutil


def get_boe_information(path):
    name = os.path.split(path)[1].split('.')[0]
    dir = os.path.split(os.path.split(os.path.split(os.path.split(path)[0])[0])[0])[1]

    return name, dir


def get_challen_information(path, mode='train'):
    if mode == 'train':
        name = os.path.split(path)[1].split('.')[0]
        dir_num = os.path.split(os.path.split(path)[0])[1].split('_')[0]
        dir_time = os.path.split(os.path.split(path)[0])[1].split('128_')[1].split('_O')[0]
    else:
        name = os.path.split(path)[1].split('.')[0]
        dir_num = os.path.split(os.path.split(path)[0])[1].split('_')[0]
        dir_time = os.path.split(os.path.split(path)[0])[1].split('128_')[1].split('_O')[0]

    return name, dir_num, dir_time


def get_cheng_information(path, mode='original'):
    if mode == 'original':
        name = os.path.split(path)[1].split('.')[0]
        dir = os.path.split(os.path.split(path)[0])[1].split('.')[0]
    else:
        name = os.path.split(path)[1].split('.')[0]
        dir = os.path.split(os.path.split(os.path.split(path)[0])[0])[1].split('.')[0]

    return name, dir
# # #####  BOE  ##### #
# data_path = 'D:/Pycharm Projects/datasets/2014_BOE_Srinivasan/Publication_Dataset'
# image_list = sorted(glob.glob(data_path + '/*/TIFFs/8bitTIFFs/*'))
# count = 0
# for image in image_list:
#     count += 1
#     print('{}/{}'.format(count, len(image_list)))
#     name, dir = get_boe_information(image)
#     shutil.copy(image, 'D:/Pycharm Projects/datasets/Source1' + '/{}-{}.jpg'.format(dir, name))

# #####  Challen   ##### #
# data_path = 'D:\Pycharm Projects\datasets\AI_OCT_Challenger'
# train_image_list = sorted(glob.glob(data_path +
#                                     '/ai_challenger_fl2018_trainingset/Edema_trainingset/original_images/*/*.bmp'))
# val_image_list = sorted(glob.glob(data_path +
#                                   '/ai_challenger_fl2018_validationset/Edema_validationset/original_images/*/*.bmp'))
# count = 0
# for image in train_image_list:
#     count += 1
#     print('{}/{}'.format(count, len(train_image_list)))
#     name, dir_num, dir_time = get_challen_information(image, mode='train')
#     shutil.copy(image, 'D:/Pycharm Projects/datasets/Target/train-{}-{}-{}.jpg'.format(dir_num, dir_time, name))
#
# count = 0
# for image in val_image_list:
#     count += 1
#     print('{}/{}'.format(count, len(val_image_list)))
#     name, dir_num, dir_time = get_challen_information(image, mode='val')
#     shutil.copy(image, 'D:/Pycharm Projects/datasets/Target/val-{}-{}-{}.jpg'.format(dir_num, dir_time, name))


# ######  cheng   ##### #
train_data_path = 'D:/Pycharm Projects/datasets/PreivateChengOCT'
image_list = sorted(glob.glob(train_data_path + '/*.fds/*.png'))
image_mask = sorted(glob.glob(train_data_path + '/*.fds/gt_10/*.png'))
count = 0
for image in image_list:
    count += 1
    print('{}/{}'.format(count, len(image_list)))
    name, dir = get_cheng_information(image)
    shutil.copy(image, 'D:/Pycharm Projects/datasets/cheng' + '/{}-{}.png'.format(dir, name))

for image in image_mask:
    count += 1
    print('{}/{}'.format(count, len(image_list)))
    name, dir = get_cheng_information(image, mode='mask')
    shutil.copy(image, 'D:/Pycharm Projects/datasets/cheng_mask' + '/{}-{}.png'.format(dir, name))
