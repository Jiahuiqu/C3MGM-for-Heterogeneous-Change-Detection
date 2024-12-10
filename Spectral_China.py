import time
import numpy as np
import scipy.io as sio
import torch
from skimage.transform import rescale
import matplotlib.pyplot as plt
from Spectral_SR.model_27 import SpeSRModel
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
import torch.nn.functional as F


name_data = 'China'
load_origin_dir = './data/China.mat'
load_patch_dir = './data/data_Spectral/' + name_data
gpu = 'cuda:1'
device = torch.device(gpu)
batch_size = 64
bands = 154
num_epoch = 150

def create_patch(load_dir, t, patchsize_HSI):
    print('===>>>处理{}时刻图片'.format(t))
    # 读入图片
    datas = sio.loadmat(load_dir)
    if(t=='t2'):
        img = np.array(datas['T2'][...], dtype=np.float32)
    elif(t=='t1'):
        img = np.array(datas['T1'][...], dtype=np.float32)
    else:
        raise Exception('No this img!')
    img = normalize_data(img)
    print('原图片尺寸:',img.shape)
    height, width, bands = img.shape[0], img.shape[1], img.shape[2]

    pan = np.mean(img[:,:,:27],axis=2)
    print('RGB图片尺寸：',pan.shape)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.mean(img, axis=2))
    axs[0].set_title('pan_old')
    axs[1].imshow(pan)
    axs[1].set_title('pan_new')
    print('./data/data_Spectral/' + name_data + '.mat')
    plt.show()

    # 切patch（多光谱）

    # 构建高光谱patch序列
    k = 0
    for i in range(int((height-27)/3)):
        for j in range(int((width-27)/3)):
            k = k + 1
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = img[(i*3):(i*3+27),
                    (j*3):(j*3+27),:]
            patch_pan = pan[(i*3):(i*3+27),
                    (j*3):(j*3+27)]
            if (patch.shape[0]!=27)or(patch.shape[1]!=27):
                raise Exception(height,i*3+27,width, j*3+27)
            sio.savemat('./data/data_Spectral/' + name_data + '/' + str(k) + '.mat', mdict={'hs':patch,'pan':patch_pan})
        if (i%20)==0:
            print('./data/data_Spectral/' + name_data + '.......' + str(i) + '/',int(height/3))
        if (i % 40 == 0):
            fig, axs = plt.subplots(1, 2)
            fig.suptitle('i*3={},j*3={}'.format(i*3, j*3))
            axs[0].imshow(np.mean(patch, axis=2))
            axs[0].set_title('pan')
            axs[1].imshow(patch_pan)
            axs[1].set_title('pan')
            plt.show()
    print('总计{}个patch已切完'.format(k))

def mat2gray(matrix):
    """将矩阵中的像素值范围映射到 [0, 1] 之间的范围"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if min_val != max_val:
        normalized = (matrix - min_val) / (max_val - min_val)
    else:
        normalized = matrix
    return normalized

def normalize_data(data):
    # 获取每个特征的最大值和最小值
    data2 = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    # min_vals = np.min(data2, axis=0)
    # max_vals = np.max(data2, axis=0)
    min_vals = np.min(data2)
    max_vals = np.max(data2)
    # 将数据减去最小值，再除以最大值和最小值的差
    normalized_data2 = (data2 - min_vals) / (max_vals - min_vals)
    normalized_data = normalized_data2.reshape(data.shape[0],data.shape[1],data.shape[2])
    return normalized_data

class DatasetSpectral(Dataset):
    def __init__(self,patch_dir):
        self.patch_files = [os.path.join(patch_dir, x) for x in os.listdir(patch_dir)]

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, index):
        load_dir = self.patch_files[index]
        data = sio.loadmat(load_dir)
        hs = np.array(data['hs'][...])
        pan = np.array(data['pan'][...])
        return hs,pan


def Spectral(load_patch_dir):
    all_dataset = DatasetSpectral(load_patch_dir)
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    nets_SpeSR = SpeSRModel(input_features=1, output_features=bands).to(device)
    print('len(all_dataset):',len(all_dataset),'len(all_dataloader):',len(all_dataloader))
    parameters = list(nets_SpeSR.parameters())
    loss1 = nn.L1Loss()
    loss2 = nn.MSELoss()
    lr = 5e-5
    beta1 = 0.9
    beta2 = 0.999
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-6, betas=(beta1, beta2))

    for epoch in range(num_epoch):
        nets_SpeSR.train()
        time1 = time.time()
        rate = int(epoch / 10)
        lr = lr * np.power(0.93, rate)
        loss_sum = 0
        for batch, (hs,pan) in enumerate(all_dataloader):
            optimizer.zero_grad()
            hs = hs.to(device).permute(0,3,1,2).float()
            pan = pan.to(device).unsqueeze(dim=1).float()
            hs_new = nets_SpeSR(pan)
            loss = loss1(hs_new,hs) + loss2(hs_new,hs)
            loss.backward()
            loss_sum = loss_sum + loss
            optimizer.step()
        print('{}/{}epochs.{}s for this epoch.average loss:{}'.format(epoch+1,num_epoch,time.time()-time1,loss_sum/len(all_dataloader)))
        with  torch.no_grad():
            if (epoch % 5 == 0):
                pan = pan.squeeze().detach().cpu().numpy()
                pan = pan[0, :, :]
                hs = hs[:, (5, 20, 40), :, :]
                hs = hs.permute(0,2,3,1).detach().cpu().numpy()
                hs = hs[0, :, :, :]
                hs_new = hs_new[:, (5, 20, 40), :, :]
                hs_new = hs_new.permute(0,2,3,1).detach().cpu().numpy()
                hs_new = hs_new[0, :, :, :]
                fig, axs = plt.subplots(1, 3)
                fig.suptitle('epoch' + str(epoch + 1) + ', show the SR:')
                axs[0].imshow(pan)
                axs[0].set_title('pan')
                axs[1].imshow(hs_new)
                axs[1].set_title('SR')
                axs[2].imshow(hs)
                axs[2].set_title('origin')
                plt.show()
            if ((epoch+1)%30==0):
                with torch.no_grad():
                    nets_SpeSR.eval()
                    torch.save(nets_SpeSR.state_dict(),
                               './Spectral_SR/pretrained_model/' + name_data + '_pretrained_' + str(epoch + 1) + '.pth')


create_patch(load_origin_dir, t='t2', patchsize_HSI=9)
Spectral(load_patch_dir)