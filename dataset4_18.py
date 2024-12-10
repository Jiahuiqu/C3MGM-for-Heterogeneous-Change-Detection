import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AllDataset_no_mid(Dataset):
    def __init__(self,patch_dir):
        # patch_folders = [os.path.join(patch_dir, x) for x in os.listdir(patch_dir)]
        # self.patch_files = []
        # for i in range(len(patch_folders)):
        #     self.patch_files.extend([os.path.join(patch_folders[i], x) for x in os.listdir(patch_folders[i])])
        self.patch_files = [os.path.join(patch_dir, x) for x in os.listdir(patch_dir)]

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, index):
        load_dir = self.patch_files[index]
        data = sio.loadmat(load_dir)
        hs_t1 = np.array(data['hs_t1'][...])
        hs_t2 = np.array(data['hs_t2'][...])
        pan_t1 = np.array(data['pan_t1'][...])
        pan_t2 = np.array(data['pan_t2'][...])
        arr = np.array(data['arr']).squeeze()
        list_gt = np.array(data['gt']).squeeze()
        pseudo_label = np.array(data['pseudo_label']).squeeze()
        # upsample_t1 = np.array(data['upsample_t1'][...])
        return hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label,index

class AllDataset224(Dataset):
    def __init__(self,patch_dir):
        # patch_folders = [os.path.join(patch_dir, x) for x in os.listdir(patch_dir)]
        # self.patch_files = []
        # for i in range(len(patch_folders)):
        #     self.patch_files.extend([os.path.join(patch_folders[i], x) for x in os.listdir(patch_folders[i])])
        self.patch_files = [os.path.join(patch_dir, x) for x in os.listdir(patch_dir)]

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, index):
        load_dir = self.patch_files[index]
        data = sio.loadmat(load_dir)
        hs_t1 = np.array(data['hs_t1'][...])
        hs_t2 = np.array(data['hs_t2'][...])
        pan_t1 = np.array(data['pan_t1'][...])
        pan_t2 = np.array(data['pan_t2'][...])
        arr = np.array(data['arr']).squeeze()
        arr2 = np.array(data['arr2']).squeeze()
        list_gt = np.array(data['gt']).squeeze()
        pseudo_label = np.array(data['pseudo_label']).squeeze()
        # upsample_t1 = np.array(data['upsample_t1'][...])
        return hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label,index,arr2