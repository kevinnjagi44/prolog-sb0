import glob
import os
from PIL import Image
import scipy.io
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import *
import pdb

transform_default = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_dataloader(dataset, root, batch_size, transform=transform_default, mode='train'):
    if dataset == 'boe':
        dataloader = DataLoader(oct_boe_dataset(root, transform=transform, mode=mode),
                                batch_size=batch_size, shuffle=True)
    elif dataset == 'cell':
        dataloader = DataLoader(oct_cell_dataset(root, transform=transform, mode=mode),
                                batch_size=batch_size, shuffle=True)
    else:
        raise ValueError('dataset not exist')

    return dataloader


class oct_cheng_frames_dataset(Dataset):
    def __init__(self, root, transform=transform_default, mode='train', source_num=4):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.source_num = source_num
        self.fragment = []
        self.volumes = sorted(glob.glob(self.root + '/*.fds'))

        for single_volume in self.volumes[:]:
            num_list = []  # frames 的序号
            image_list_sorted = []  # real world image 排序

            images = sorted(glob.glob(single_volume + '/*.png'))
            for i in images:
                num_list.append(int(i.split('oct')[1].split('.')[0]))
            num_sort = np.argsort(num_list)
            for i in range(len(num_sort)):
                image_list_sorted.append(images[num_sort[i]])
            for i in range(len(image_list_sorted) - self.source_num):
                self.fragment.append(image_list_sorted[i: i + self.source_num + 1])

    def __getitem__(self, index):
        for i in range(len(self.fragment[index]) - 1):
            image = self.transform(Image.open(self.fragment[index][i]).convert('L'))
            if i == 0:
                output = image
            else:
                output = torch.cat((output, image), 0)
        target = self.transform(Image.open(self.fragment[index][self.source_num]).convert('L'))

        return output, target

    def __len__(self):
        return len(self.fragment)


class oct_boe_and_challen_dataset(Dataset):
    def __init__(self, root1, root2, transform=transform_default, mode='val', source_num=4):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        self.mode = mode
        self.source_num = source_num
        self.fragment = []
        self.normal_volumes = sorted(glob.glob(self.root1 + '/NORMAL*'))
        self.EDEMA_volumes = sorted(glob.glob(self.root2 + '/original_images/P*'))

        with open(os.path.join(self.root2, 'label_images/EDEMA_label.pkl'), 'rb') as fr:
            self.EDEMA_label_dict = pickle.load(fr)

        if self.mode == 'val':
            self.normal_volumes = self.normal_volumes[:int(len(self.normal_volumes) / 2)]
            self.EDEMA_volumes = self.EDEMA_volumes[:int(len(self.EDEMA_volumes) / 2)]
        elif self.mode == 'test':
            self.normal_volumes = self.normal_volumes[int(len(self.normal_volumes) / 2):]
            self.EDEMA_volumes = self.EDEMA_volumes[int(len(self.EDEMA_volumes) / 2):]
        self.volumes = self.normal_volumes + self.EDEMA_volumes

        for single_volume in self.volumes[:]:
            num_list = []  # frames 的序号
            image_list_sorted = []  # real world image 排序

            images = sorted(glob.glob(single_volume + '/TIFFs/8bitTIFFs/*.tif'))
            if len(images) == 0:
                images = sorted(glob.glob(single_volume + '/*'))
            for i in images:
                num_list.append(int(os.path.split(i)[-1].split('.')[0]))

            num_sort = np.argsort(num_list)
            for i in range(len(num_sort)):
                image_list_sorted.append(images[num_sort[i]])
            for i in range(len(image_list_sorted) - self.source_num):
                self.fragment.append(image_list_sorted[i: i + self.source_num + 1])

    def __getitem__(self, index):
        for i in range(len(self.fragment[index]) - 1):
            image = self.transform(Image.open(self.fragment[index][i]).convert('L'))
            if i == 0:
                output = image
            else:
                output = torch.cat((output, image), 0)
        target_image = self.transform(Image.open(self.fragment[index][self.source_num]).convert('L'))
        target_path = self.fragment[index][self.source_num]
        target_name = os.path.split(target_path)[-1]
        if target_name.find('tif') != -1:
            label = 0
        else:
            label_dir = os.path.split(os.path.split(target_path)[0])[-1].split('.')[0]
            label = self.EDEMA_label_dict[label_dir + '_labelMark'][target_name]

        return output, target_image, label

    def __len__(self):
        return len(self.fragment)


class oct_boe_frames_dataset(Dataset):
    # 每个volume采样连续的几张图像
    def __init__(self, root, transform=transform_default, mode='val', source_num=4):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.source_num = source_num
        self.fragment = []
        self.normal_volumes = sorted(glob.glob(self.root + '/NORMAL*'))
        self.amd_volumes = sorted(glob.glob(self.root + '/AMD*'))
        self.dme_volumes = sorted(glob.glob(self.root + '/DME*'))

        if self.mode == 'train':
            self.normal_volumes = self.normal_volumes[:5]
            self.amd_volumes = self.amd_volumes[:5]
            self.dme_volumes = self.dme_volumes[:5]
        elif self.mode == 'val':
            self.normal_volumes = self.normal_volumes[5:10]
            self.amd_volumes = self.amd_volumes[5:10]
            self.dme_volumes = self.dme_volumes[5:10]
        elif self.mode == 'test':
            self.normal_volumes = self.normal_volumes[10:]
            self.amd_volumes = self.amd_volumes[10:]
            self.dme_volumes = self.dme_volumes[10:]
        self.volumes = self.normal_volumes + self.amd_volumes + self.dme_volumes

        for single_volume in self.volumes[:]:
            num_list = []  # frames 的序号
            image_list_sorted = []  # real world image 排序

            images = sorted(glob.glob(single_volume + '/TIFFs/8bitTIFFs/*.tif'))
            for i in images:
                num_list.append(int(os.path.split(i)[-1].split('.')[0]))

            num_sort = np.argsort(num_list)
            for i in range(len(num_sort)):
                image_list_sorted.append(images[num_sort[i]])
            for i in range(len(image_list_sorted) - self.source_num):
                self.fragment.append(image_list_sorted[i: i + self.source_num + 1])

    def __getitem__(self, index):
        for i in range(len(self.fragment[index]) - 1):
            image = self.transform(Image.open(self.fragment[index][i]).convert('L'))
            if i == 0:
                output = image
            else:
                output = torch.cat((output, image), 0)
        target_image = self.transform(Image.open(self.fragment[index][self.source_num]).convert('L'))
        target_path = self.fragment[index][self.source_num]
        target_name = os.path.split(target_path)[-1]
        if target_name.find('tif') != -1:
            label = 0
        else:
            label_dir = os.path.split(os.path.split(target_path)[0])[-1].split('.')[0]
            label = self.EDEMA_label_dict[label_dir + '_labelMark'][target_name]

        return output, target_image, label

    def __len__(self):
        return len(self.fragment)


class boe_dataset_zhoukang_test(Dataset):
    def __init__(self, root, transform=transform_default, mode='val'):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.image_list = []
        self.normal_volumes = sorted(glob.glob(self.root + '/NORMAL*'))
        self.amd_volumes = sorted(glob.glob(self.root + '/AMD*'))
        self.dme_volumes = sorted(glob.glob(self.root + '/DME*'))

        self.volumes = self.normal_volumes + self.amd_volumes + self.dme_volumes

        for single_volume in self.volumes[:]:
            images = sorted(glob.glob(single_volume + '/TIFFs/8bitTIFFs/*.tif'))
            self.image_list = self.image_list + images

    def __getitem__(self, index):
        output = self.transform(Image.open(self.image_list[index]).convert('L'))

        return output

    def __len__(self):
        return len(self.image_list)


class oct_cheng_dataset(Dataset):
    def __init__(self, root, transforms_=transform_default):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(root + '/*.fds/*.png'))
        self.gt = sorted(glob.glob(root + '/*.fds/gt_10/*.png'))

    def __getitem__(self, index):

        img = Image.open(self.files[index]).convert('L')
        gt = Image.open(self.gt[index]).convert('L')

        # if np.random.random() < 0.5:
        #     img = Image.fromarray(np.array(img)[:, ::-1, :], 'RGB')
        #     gt = Image.fromarray(np.array(gt)[:, ::-1, :], 'RGB')

        img = self.transform(img)
        image2tensor = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                           transforms.ToTensor()])
        gt = image2tensor(gt)

        return {'A': img, 'B': gt}

    def __len__(self):
        return len(self.files)


class oct_edema_dataset(Dataset):
    def __init__(self, root, transforms_=transform_default):
        self.transform = transforms.Compose(transforms_)
        self.path = root + '/ai_challenger_fl2018_trainingset/original_images/*/*.bmp'
        self.files = sorted(glob.glob(root + '/ai_challenger_fl2018_trainingset/original_images/*/*.bmp'))
        self.gt = sorted(glob.glob(root + '/ai_challenger_fl2018_trainingset/label_images/*/*.bmp'))

    def __getitem__(self, index):

        img = Image.open(self.files[index]).convert('L')
        gt = Image.open(self.gt[index]).convert('L')

        # if np.random.random() < 0.5:
        #     img = Image.fromarray(np.array(img)[:, ::-1, :], 'RGB')
        #     gt = Image.fromarray(np.array(gt)[:, ::-1, :], 'RGB')

        img = self.transform(img)
        image2tensor = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                           transforms.ToTensor()])
        gt = image2tensor(gt)

        return {'A': img, 'B': gt}

    def __len__(self):
        return len(self.files)


class oct_boe_dataset(Dataset):
    def __init__(self, root, transform=transform_default, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.image_label = []
        self.normal_volumes = sorted(glob.glob(self.root + '/NORMAL*'))
        self.amd_volumes = sorted(glob.glob(self.root + '/AMD*'))
        self.dme_volumes = sorted(glob.glob(self.root + '/DME*'))
        self.volumes = {}
        self.image_label = torch.Tensor([])
        self.image_list = []
        if mode == 'train':
            for i in self.normal_volumes[:9]:
                self.volumes[i] = 0
            for i in self.amd_volumes[:9]:
                self.volumes[i] = 1
            # self.volumes = self.normal_volumes[:12] + self.amd_volumes[:12]
        elif mode == 'val':
            # self.volumes = self.normal_volumes[12:] + self.amd_volumes[12:] + self.dme_volumes[12:]
            for i in self.normal_volumes[9:12]:
                self.volumes[i] = 0
            for i in self.amd_volumes[9:12]:
                self.volumes[i] = 1
            for i in self.dme_volumes[9:12]:
                self.volumes[i] = 2
        elif mode == 'test':
            for i in self.normal_volumes[12:]:
                self.volumes[i] = 0
            for i in self.amd_volumes[12:]:
                self.volumes[i] = 1
            for i in self.dme_volumes[12:]:
                self.volumes[i] = 2

        for single_volume in self.volumes.items():
            images = sorted(glob.glob(single_volume[0] + '/TIFFs/8bitTIFFs/*.tif'))
            self.image_list = self.image_list + images
            self.image_label = torch.cat([self.image_label, torch.ones(len(images)) * single_volume[1]])

    def __getitem__(self, index):
        output = self.transform(Image.open(self.image_list[index]))   # 分类用RGB 3通道
        label = self.image_label[index]

        return output, label

    def __len__(self):
        return len(self.image_list)


class oct_cell_dataset(Dataset):
    def __init__(self, root, transform=transform_default, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.image_label = []
        self.volumes = {}
        self.image_label = torch.Tensor([])
        self.image_list = []
        self.normal = sorted(glob.glob(os.path.join(self.root, self.mode, 'NORMAL/*.jpeg')))[:15000]
        self.dme = sorted(glob.glob(os.path.join(self.root, self.mode, 'DME/*.jpeg')))
        if mode == 'test' or mode == 'val':
            self.unknown = sorted(glob.glob(os.path.join(self.root, self.mode, 'DRUSEN/*.jpeg')))
        elif mode == 'train_unknown':
            self.unknown = sorted(glob.glob(os.path.join(self.root, self.mode, 'unknown/*.jpeg')))
        else:
            self.unknown = []

        self.image_list = self.normal + self.dme + self.unknown
        self.image_label = torch.cat([torch.zeros(len(self.normal)), torch.ones(len(self.dme)),
                                      torch.ones(len(self.unknown)) * 2])

    def __getitem__(self, index):
        output = self.transform(Image.open(self.image_list[index]).convert('L'))
        label = self.image_label[index]

        return output, label

    def __len__(self):
        return len(self.image_list)
