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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def print_args(args):
    print('\n', '*' * 30, 'Args', '*' * 30)
    print('Args: \n{}\n'.format(args))


def adjust_lr(lr, optimizer, epoch, e_freq=50):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // e_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_per_class(output, target, cls_list_name):
    acc_per_cls = {}
    acc_list = []
    cls_list = []
    if cls_list_name == 'boe':
        cls_dict = {'normal': 0, 'amd': 1, 'unknown': 2}
    else:
        cls_dict = {'normal': 0, 'dme': 1, 'unknown': 2}
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    for name in cls_dict.items():
        cls = name[1]
        acc_per_cls[name[0]] = torch.sum((pred == target.long()) & (target == cls)).float() / torch.sum(target == cls).float()
    for i in acc_per_cls.items():
        acc_list.append(i[1])
        cls_list.append(i[0])

    return cls_list, acc_list


def save_ckpt(version, state, epoch, is_best=False, project='baseline'):
    v_split_list = version.split('_')
    v_major = v_split_list[0]
    v_minor = v_split_list[1]

    ckpt_dir = os.path.join('checkpoints/', version)
    os.makedirs(ckpt_dir, exist_ok=True)
    version_filename = '{}_{}ckpt.pth.tar'.format(version, project)
    version_file_path = os.path.join(ckpt_dir, version_filename)
    torch.save(state, version_file_path)
    # if epoch % 10 == 0:
    #     ckpt_file_path = os.path.join(ckpt_dir, '{}_ckpt@{}.pth.tar'.format(version, epoch))
    #     torch.save(state, ckpt_file_path)
    if is_best:
        best_file_path = os.path.join(ckpt_dir, '{}_{}_best@{}.pth.tar'.format(v_major, v_minor, epoch))
        # shutil.copyfile(version_file_path, best_file_path)  maintain the best model
        torch.save(state, best_file_path)

