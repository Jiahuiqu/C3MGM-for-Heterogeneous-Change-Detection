import math
import time
import random
import numpy as np
import scipy.io as sio
import data.create_data,data.create_data_USA
import nets_try as nets
from torch.optim import Adam
import torch
from torch_geometric.data import Batch
import graph_matching
from utils_try import adjust_learning_rate,build_graph_lower,build_graph_upper,build_graph_middle
import matplotlib.pyplot as plt
from Spatial_SR.common import default_conv
from Spatial_SR.SSPSR import SSPSR
from Spectral_SR.model_27 import SpeSRModel,SpeSRModel224
from torch.utils.data import DataLoader
from dataset4_18 import AllDataset_no_mid as AllDataset

gpu = 'cuda:0'
device = torch.device(gpu)

model_name_Spectral = './Spectral_SR/pretrained_model/USA_pretrained_150.pth' # 预训练光谱超分模型
load_origin_dir = './data/China.mat'

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
    min_vals = np.min(data2, axis=0)
    max_vals = np.max(data2, axis=0)
    # 将数据减去最小值，再除以最大值和最小值的差
    normalized_data2 = (data2 - min_vals) / (max_vals - min_vals)
    normalized_data = normalized_data2.reshape(data.shape[0],data.shape[1],data.shape[2])
    return normalized_data


def main():
    datas = sio.loadmat(load_origin_dir)
    img = np.array(datas['T2'][...], dtype=np.float32)
    img = normalize_data(img)
    height, width, bands = img.shape[0], img.shape[1], img.shape[2]
    img_pan = np.mean(img[:,:,:27], axis=2)
    # img_pan = mat2gray(img_pan)
    img_pan = np.pad(img_pan, ((13,13),(13,13)), 'symmetric')
    print(img_pan.shape)


    nets_SpeSR = SpeSRModel(input_features=1, output_features=bands).to(device)
    state_dict2 = torch.load(model_name_Spectral)
    nets_SpeSR.load_state_dict(state_dict2)

    with torch.no_grad():
        nets_SpeSR.eval()
        img_new = np.zeros((height,width,bands))
        for i in range(int(height)):
            if i % 10 == 0:
                print('{}/{}'.format(i,int(height)))
                time1 = time.time()
            for j in range(int(width)):
                patch = img_pan[i:i+27,j:j+27]
                patch = torch.from_numpy(patch).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
                patch = nets_SpeSR(patch)
                patch = patch.permute(0,2,3,1).squeeze()
                img_new[i,j,:] = patch[13,13,:].detach().cpu().numpy()

            if (i+1) % 10 == 0:
                print('time for 50i:{}'.format(time.time()-time1))
        plt.figure()
        plt.imshow(img_new[:, :, (5, 15, 30)])
        plt.show()
        sio.savemat("./data/new_from_USA_to_China_5_27.mat",mdict={'hs':img_new})


main()