import numpy as np
import scipy.io as sio
import scipy
from skimage.transform import rescale
import torch
from scipy.io import loadmat, savemat
from torch import nn
import torchvision.transforms as transforms
import time


def create_pre_training_data(t1_img):
    print('===> start create pre_training data')
    time1 = time.time()

    data_HSI = t1_img
    patchsize_HSI = 27

    height, width, bands = data_HSI.shape
    # ##### 给HSI和LiDAR打padding #####
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = data_HSI[:, :, 0]   #141x48
    pad_width = np.floor(patchsize_HSI / 2) #9/2 = 4
    pad_width = np.int32(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    print('img shape after padding:',temp2.shape)
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = data_HSI[:, :, i]
        pad_width = np.floor(patchsize_HSI / 2)
        pad_width = np.int32(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        data_HSI_pad[:, :, i] = temp2

    # #### 构建高光谱的训练集 ####
    TrainNum = width*height
    # TrainPatch_gt = np.empty((TrainNum, patchsize_HSI, patchsize_HSI, bands), dtype='float32') # 27x27
    # TrainPatch_ms = np.empty((TrainNum, int(patchsize_HSI/3), int(patchsize_HSI/3), bands), dtype='float32') # 9x9
    # TrainPatch_lms = np.empty((TrainNum, patchsize_HSI, patchsize_HSI, bands), dtype='float32') # 27x27
    # print('shape of the pre_training img:<gt,ms,lms> :<',TrainPatch_gt.shape,TrainPatch_ms.shape,TrainPatch_lms.shape,'>')

    k = 0
    for i in range(height):
        for j in range(width):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = data_HSI_pad[(i):(i + 2 * pad_width + 1),
                    (j):(j + 2 * pad_width + 1), :]
            # TrainPatch_gt[k, :, :, :] = patch
            patch_ms = rescale(patch, ( 1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
            # TrainPatch_ms[k, :, :, :] = patch_ms
            patch_lms = rescale(patch_ms, (3, 3, 1), order=3, mode='edge', anti_aliasing=True)
            # TrainPatch_lms[k, :, :, :] = patch_lms

            if((k-3)%10==0): #k-3是随便给的,每隔10个一个验证集
                scipy.io.savemat('./cut_data/pre_training_China/evals/pre_train_data_' + str(k) + '.mat',
                                 mdict = {'gt': patch, 'ms': patch_ms, 'lms': patch_lms, 'num': k})
            elif((k-8)%10==0): #k-8是随便给的,每隔10个一个测试集
                scipy.io.savemat('./cut_data/pre_training_China/tests/pre_train_data_' + str(k) + '.mat',
                                 mdict = {'gt': patch, 'ms': patch_ms, 'lms': patch_lms, 'num': k})
            else:
                scipy.io.savemat('./cut_data/pre_training_China/trains/pre_train_data_' + str(k) + '.mat',
                                 mdict={'gt': patch, 'ms': patch_ms, 'lms': patch_lms, 'num': k})
            k = k + 1
            if (k%300==0):
                print(k, '个patch已切,共有',TrainNum,'个')
                time1 = time.time() - time1
                print('生成300个patch耗时{:.2f}秒', time1)
                time1 = time.time()
    print('总计',k,'个patch已切完,successfully created train_data')

def create_org_img(load_dir,create_mat):
    # 生成真值为423x144，ms为141x48的original_img
    datas = sio.loadmat(load_dir)
    t1_img = np.array(datas['T1'][...], dtype=np.float32)
    t1_img = normalize_data(t1_img)
    print('原始chinaT1图片的尺寸 :', t1_img.shape)
    height, width, bands = t1_img.shape
    temp = t1_img[:, :, 0]  # 420x140
    temp2 = np.pad(temp, ((1, 2), (2, 2)), 'symmetric')
    [h_pad, w_pad] = temp2.shape
    t1_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = t1_img[:, :, i]
        temp2 = np.pad(temp, ((1, 2), (2, 2)), 'symmetric')
        t1_pad[:, :, i] = temp2
    ms = rescale(t1_pad, (1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
    lms = rescale(ms, (3, 3, 1), order=3, mode='edge', anti_aliasing=True)
    if create_mat == True:
        scipy.io.savemat('./cut_data/pre_training_China/origin/orl_img' + '.mat', mdict={'gt': t1_pad, 'ms': ms, 'lms': lms})
    return ms,t1_pad

# 对高光谱数据进行最大最小值归一化
def normalize_data(data):
    # 获取每个特征的最大值和最小值
    data2 = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    min_vals = np.min(data2, axis=0)
    max_vals = np.max(data2, axis=0)
    # 将数据减去最小值，再除以最大值和最小值的差
    normalized_data2 = (data2 - min_vals) / (max_vals - min_vals)
    normalized_data = normalized_data2.reshape(data.shape[0],data.shape[1],data.shape[2])
    return normalized_data

def cut_data(load_dir):
    img_lr,img_hr = create_org_img(load_dir,create_mat = True)
    print('原始chinaT1图片填充后尺寸：',img_hr.shape)
    print('超分前chinaT1图片尺寸：',img_lr.shape)
    create_pre_training_data(img_lr)








    # #### 数据转换以及把数据搬到GPU #####
    # device = torch.device('cuda')
    # TrainPatch_gt = torch.from_numpy(TrainPatch_gt).to(device)
#
# load_dir = '/run/media/xd132/E/YHD/X师姐的网/SSPSR-master/dataset/China.mat'
# cut_data(load_dir)

