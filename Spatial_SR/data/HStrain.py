import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import utils


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_folders = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_files = []

        self.image_files = self.image_folders

        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        # print('index.shape',index.shape,index)
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        ms = np.array(data['ms'][...], dtype=np.float32)
        lms = np.array(data['lms'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)
        num = np.array(data['num'])
        ms, lms, gt, num = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num), utils.data_augmentation(num, mode=aug_num)
        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
            num = torch.from_numpy(num.copy())
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            num = torch.from_numpy(num.copy())
        return ms, lms, gt, num

    def __len__(self):
        return len(self.image_files)*self.factor


class HSEvalData(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_folders = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_files = []
        self.image_files = self.image_folders
        # for i in self.image_folders:
        #
        #     images = i
        #     for j in images:
        #         if is_mat_file(j):
        #             full_path = os.path.join(i, j)
        #             self.image_files.append(full_path)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        ms = np.array(data['ms'][...], dtype=np.float32)
        lms = np.array(data['lms'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)
        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num)
        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)*self.factor