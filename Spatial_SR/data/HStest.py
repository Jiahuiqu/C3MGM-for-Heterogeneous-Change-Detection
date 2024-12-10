import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os


class HSTestData_one_img(data.Dataset):
    def __init__(self, image_dir, use_3D=False):
        test_data = sio.loadmat(image_dir)
        self.use_3Dconv = use_3D
        self.ms = np.array(test_data['ms'][...], dtype=np.float32)
        # self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
        self.lms = np.array(test_data['lms'][...], dtype=np.float32)
        self.gt = np.array(test_data['gt'][...], dtype=np.float32)

    def __getitem__(self, index):
        # gt = self.gt[index, :, :, :]
        # ms = self.ms[index, :, :, :]
        # lms = self.lms[index, :, :, :]
        gt = self.gt
        ms = self.ms
        lms = self.lms
        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
        #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
        #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
        return ms, lms, gt

    def __len__(self):
        return 1


class HSTestData_many_img(data.Dataset):
    def __init__(self, image_dir, use_3D=False):

        self.image_folders = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_files = []
        self.image_files = self.image_folders

        self.use_3Dconv = use_3D
        # self.ms = np.array(test_data['ms'][...], dtype=np.float32)
        # # self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
        # self.lms = np.array(test_data['lms'][...], dtype=np.float32)
        # self.gt = np.array(test_data['gt'][...], dtype=np.float32)

    def __getitem__(self, index):

        file_index = index
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        ms = np.array(data['ms'][...], dtype=np.float32)
        lms = np.array(data['lms'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)
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




        # gt = self.gt[index, :, :, :]
        # ms = self.ms[index, :, :, :]
        # lms = self.lms[index, :, :, :]
        # if self.use_3Dconv:
        #     ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
        #     ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
        #     lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
        #     gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        # else:
        #     ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        #     lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        #     gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        # #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
        # #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
        # #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
        # return ms, lms, gt

    def __len__(self):
        return len(self.image_files)
