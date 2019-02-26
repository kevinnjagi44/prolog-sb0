import os
import torch
import torchvision.transforms as transforms

import pdb


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0.0)


def image_saver(path, image_tensor, count, num_channel=1):
    transform = transforms.ToPILImage()

    assert image_tensor.size()[1] == num_channel
    N = image_tensor.size()[0]
    image = transform(torch.cat([image_tensor[N-2], image_tensor[N-1]], 2).cpu())
    image.save(os.path.join(path, str(count).zfill(6)) + '.jpg')


class Counter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0

    def step(self):
        self.count += 1


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
