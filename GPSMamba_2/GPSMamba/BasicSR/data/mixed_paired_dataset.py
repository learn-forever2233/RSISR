import random
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np


@DATASET_REGISTRY.register()
class MixedPairedImageDataset(data.Dataset):
    """Mixed Paired image dataset with dynamic LQ selection (bicubic or Gaussian).

    This dataset supports two separate LQ-GT pairs (bicubic and Gaussian), and dynamically selects
    between them for each sample based on a probability threshold.

    Args:
        opt (dict): Config for train datasets. Contains the following keys:
            dataroot_gt_bicubic (str): Data root path for bicubic GT.
            dataroot_lq_bicubic (str): Data root path for bicubic LQ.
            dataroot_gt_gaosi (str): Data root path for Gaussian GT.
            dataroot_lq_gaosi (str): Data root path for Gaussian LQ.
            gaussian_prob (float): Probability (0-1) of using Gaussian LQ. Default: 0.2.
            ... (other keys same as PairedImageDataset)
    """

    def __init__(self, opt):
        super(MixedPairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gaussian_prob = opt.get('gaussian_prob', 0.2)

        self.gt_bicubic_folder = opt['dataroot_gt_bicubic']
        self.lq_bicubic_folder = opt['dataroot_lq_bicubic']
        self.gt_gaosi_folder = opt['dataroot_gt_gaosi']
        self.lq_gaosi_folder = opt['dataroot_lq_gaosi']

        if 'filename_tmpl_bicubic' in opt:
            self.filename_tmpl_bicubic = opt['filename_tmpl_bicubic']
        elif 'filename_tmpl' in opt:
            self.filename_tmpl_bicubic = opt['filename_tmpl']
        else:
            self.filename_tmpl_bicubic = '{}'

        if 'filename_tmpl_gaosi' in opt:
            self.filename_tmpl_gaosi = opt['filename_tmpl_gaosi']
        else:
            self.filename_tmpl_gaosi = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError('LMDB not supported for MixedPairedImageDataset')
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            raise NotImplementedError('Meta info file not supported for MixedPairedImageDataset')
        else:
            self.paths_bicubic = paired_paths_from_folder(
                [self.lq_bicubic_folder, self.gt_bicubic_folder], ['lq', 'gt'], self.filename_tmpl_bicubic, self.task)
            self.paths_gaosi = paired_paths_from_folder(
                [self.lq_gaosi_folder, self.gt_gaosi_folder], ['lq', 'gt'], self.filename_tmpl_gaosi, self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        use_gaosi = random.random() < self.gaussian_prob
        if use_gaosi:
            paths = self.paths_gaosi
            lq_type = 'gaosi'
        else:
            paths = self.paths_bicubic
            lq_type = 'bicubic'

        actual_index = index % len(paths)

        gt_path = paths[actual_index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = paths[actual_index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'lq_type': lq_type}

    def __len__(self):
        return max(len(self.paths_bicubic), len(self.paths_gaosi))
