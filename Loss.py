import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()

    def forward(self, input, target):
        _assert_no_grad(target)
        col_grad_input = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        col_grad_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        row_grad_input = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        row_grad_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        return F.l1_loss(col_grad_input, col_grad_target) + F.l1_loss(row_grad_input, row_grad_target)

