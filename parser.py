# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       parser_v01
   Project Name:    segOCT
   Author :         Kang ZHOU
   Date:            2018/11/12
-------------------------------------------------
   Change Activity:
                   2018/11/12:
-------------------------------------------------
"""
import argparse
import warnings


class ParserArgs(object):
    def __init__(self):
        parser = self.get_parser()
        args = parser.parse_args()
        args.node =self.get_node()

        self.assert_version(args.version)

        self.args = args

    def get_parser(self):
        parser = argparse.ArgumentParser(
            description='PyTorch Segmentation Training and Testing'
        )
        # general useful args
        parser.add_argument('--version', help='the version of different method/setting/parameters etc')
        # dataset
        parser.add_argument('--dataset',
                            default='boe',
                            help='dataset name')
        parser.add_argument('--data_path',
                            default='/root/datasets/CellOCT2017',
                            help='path to dataset')
        parser.add_argument('--sina_data',
                            default='/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/',
                            help='path to sina dataset')
        parser.add_argument('--edema_data',
                            default='/root/datasets/OCT_challenge',
                            help='path to edema dataset')
        parser.add_argument('--cheng_data',
                            default='/root/datasets/OCT_cheng',
                            help='path to cheng dataset')
        parser.add_argument('--boe_data',
                            default='/root/datasets/OCT_BOE',
                            help='path to cell dataset')
        parser.add_argument('--cell_data',
                            default='/root/datasets/CellOCT2017',
                            help='path to boe dataset')
        # retrain
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--mode', default='', type=str, metavar='PATH',
                            help='checkpoint mode (default: none)')
        parser.add_argument('--validate', action='store_true',
                            help='resume and validate')

        # model hyper-parameters
        parser.add_argument('--epochs', default=240, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--batch-size', default=64, type=int,
                            metavar='N', help='mini-batch size')
        parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')

        # experiments
        parser.add_argument('--test-transfer', action='store_true',
                            help='use transfered test data(default)')

        # visdom
        parser.add_argument('--port', default=31670, type=int, help='visdom port')

        # other useful args
        parser.add_argument('--gpu', nargs='+', type=int, help='gpu id for single/multi gpu')
        parser.add_argument('--vis-freq', default=10, type=int,
                            help='data sent frequency to visdom server')
        parser.add_argument('--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 90)')
        parser.add_argument('--project', default='2018_OCT_transfer',
                            help='project name in workspace')

        return parser

    @staticmethod
    def get_node():
        import socket
        return socket.gethostname()

    @staticmethod
    def assert_version(version):
        v_split_list = version.split('_')
        v_len = len(v_split_list) == 3
        v_major = v_split_list[0][0] == 'v' and v_split_list[0][1:].isdigit() and len(v_split_list[0]) == 3
        v_minor = v_split_list[1].isdigit() and len(v_split_list[1]) == 2
        assert v_major and v_minor, 'The version name is wrong'
        if not v_len:
            warnings.warn('The version name is warning')
