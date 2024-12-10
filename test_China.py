import os
import random
import math
import time
from random import sample
import numpy as np
import scipy.io as sio
import data.create_data,data.create_data_USA
import nets_5_31 as nets
from torch.optim import Adam
import torch
from torch_geometric.data import Batch
import graph_matching_noS as graph_matching
from utils_try import adjust_learning_rate,build_graph_lower,build_graph_upper,build_graph_middle
import matplotlib.pyplot as plt
from Spatial_SR.common import default_conv
from Spatial_SR.SSPSR import SSPSR
from torch.utils.data import DataLoader,Subset
from HybridLoss import HybridLoss
from Spectral_SR.model_27 import SpeSRModel
from dataset4_18 import AllDataset_no_mid as AllDataset

need_attention = False
model = 0
batch_size = 64
channels = 154
net_up_size1 = 64  # 上支路第一个MLP的输出尺寸 64 128
net_up_size2 = 128  # 上支路第一个MLP的隐层尺寸 128 256
net_down_size1 = 64 # 下支路第一个CNN的输出尺寸,GCN的输入尺寸 64
mid_hidden_size = 32 # 中路隐层尺寸
patchsize_HSI = 9
device = 'cuda:0'

model_name = ''
model_dir = './save_model/China/' + model_name

def main():
    time1 = time.time()
    name_data = 'China'
    load_origin_dir = './data/China.mat'
    load_patch_dir = 'data/data_change/' + name_data
    loca = time.strftime('%Y-%m-%d')
    all_dataset = AllDataset(load_patch_dir)
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    gt, height = data.create_data.create_gt(load_origin_dir)
    width = int(len(gt) / height)

    nets_lower = [nets.CNN(net_size1=net_up_size1),nets.GCN(net_size1=net_up_size1),nets.message_net(),nets.MLP(),nets.MLP2(),nets.MLP3()]
    i = 0
    for net in nets_lower:
        state_dict = torch.load(model_dir + '/low/' + str(i) + '.pth')
        i = i + 1
        net.load_state_dict(state_dict)
        net.to(device).eval()

    nets_upper = [nets.MLP0(channels=channels, net_size1=net_up_size1,net_size2=net_up_size2),
                  nets.GCNup(patchsize=patchsize_HSI), nets.message_netup(), nets.MLPup(), nets.MLP2up(), nets.MLP3()]
    i = 0
    for net in nets_upper:
        state_dict = torch.load(model_dir + '/up/' + str(i) + '.pth')
        i = i + 1
        net.load_state_dict(state_dict)
        net.to(device).eval()

    nets_middle = [
        nets.CNNmid(channel=channels, patchsize=patchsize_HSI, down_size=net_down_size1, out_size=net_up_size1,
                        hidden_size=mid_hidden_size),
        nets.GCNmid(), nets.linear_difference(), nets.linear_classification(),nets.linear_attention()]  # 阿巴阿巴阿巴巴
    i = 0
    for net in nets_middle:
        state_dict = torch.load(model_dir + '/middle/' + str(i) + '.pth')
        i = i + 1
        net.load_state_dict(state_dict)
        net.to(device).eval()

    # 空间超分网
    nets_SSPSR = SSPSR(n_subs=8, n_ovls=1, n_colors=channels, n_blocks=3, n_feats=256,
            n_scale=3, res_scale=0.1, use_share=True, conv=default_conv)
    state_dict = torch.load(model_dir + '/SSPSR.pth')
    nets_SSPSR.load_state_dict(state_dict)
    print('use pre_trained SSPSR model name:', model_dir + '/SSPSR.pth')
    nets_SSPSR.to(device).eval()

    # 光谱超分网
    nets_SpeSR = SpeSRModel(input_features=1, output_features=channels)
    state_dict = torch.load(model_dir + '/SpeSR.pth')
    nets_SpeSR.load_state_dict(state_dict)
    print('use pre_trained SpeSR model name:', model_dir + '/SpeSR.pth')
    nets_SpeSR.to(device).eval()

    print('预加载用时：',time.time()-time1)

    time2 = time.time()
    with torch.no_grad():
        correct_sum = 0
        distances = torch.zeros(len(all_dataset))  # 中路batch间欧氏距离
        result = np.zeros(height * width)  # 分类结果
        gt_for_show = np.zeros(height * width)  # 用于展示的标签

        for batch, (hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label, _) in enumerate(
                all_dataloader):
            dif, correct, result, gt_for_show, distances, preds_2 = batch_process(
                'no_grad', batch, hs_t1, hs_t2, pan_t1, pan_t2,
                nets_upper, nets_lower, nets_middle, nets_SSPSR, nets_SpeSR,
                labels=pseudo_label, list_gt=list_gt, arr=arr, result=result, gt=gt_for_show,
                distances=distances, e=0)
            correct_sum = correct_sum + correct
        result = np.reshape(result, (height, width))
        gt = np.reshape(gt, (height, width))
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('accuracy:' + str(correct_sum / len(all_dataset)))
        axs[0].imshow(gt, cmap='gray')
        axs[0].set_title('gt')
        axs[1].imshow(result, cmap='gray')
        axs[1].set_title('result')
        plt.show()
        print('全图过程用时：',time.time()-time2)
        sio.savemat('./result/result_China/NEW_' + model_name + '.mat',
                    mdict={'e':result})

def batch_process(grad,batch,hs_t1,hs_t2,pan_t1,pan_t2,nets_upper,nets_lower,nets_middle,nets_SSR,nets_SpeSR,
                  labels=None,loss_classification=None,loss_MSE=None,loss_L1=None,optimizer=None,list_gt=None,arr=None,result=None,
                  gt=None,distances=None,e=None,arr_append0=None,arr_append1=None,index=None,ipl1=None):

    # 最后一个batch的batch_num可能会小于batch_size，其余时刻相等
    patch_num = hs_t1.shape[0]

    Sum = [0, 0, 0, 0]

    if grad=='with_grad' and ((arr_append1 is not None)or(arr_append0 is not None)):

        # 我明白啦！我明白啦！我终于明白啦！

        for i in range(len(index)):
            if index[i] in arr_append0:
                if (labels[i] == 0) and (list_gt[i, 1] == 0):
                    Sum[0] = Sum[0] + 1
                elif (labels[i] == 1) and (list_gt[i, 1] == 0):
                    Sum[1] = Sum[1] + 1
                elif (labels[i] == 0) and (list_gt[i, 1] == 1):
                    Sum[3] = Sum[3] + 1
                elif (labels[i] == 1) and (list_gt[i, 1] == 1):
                    Sum[2] = Sum[2] + 1
                labels[i] = 0
            if index[i] in arr_append1:
                if (labels[i] == 1) and (list_gt[i, 1] == 1):
                    Sum[0] = Sum[0] + 1
                elif (labels[i] == 0) and (list_gt[i, 1] == 1):
                    Sum[1] = Sum[1] + 1
                elif (labels[i] == 1) and (list_gt[i, 1] == 0):
                    Sum[3] = Sum[3] + 1
                elif (labels[i] == 0) and (list_gt[i, 1] == 0):
                    Sum[2] = Sum[2] + 1
                labels[i] = 1

    # 上路
    batch_t1_up_no_reshape = hs_t1
    batch_t1_up = torch.reshape(hs_t1,(-1,channels)).to(device)
    # batch_t2_up = torch.reshape(hs_t2,(-1,channels)).to(device)

    # batch_t1_up = torch.reshape(hs_t1[:,1:-1,1:-1,:],(-1,channels)).to(device) # 4.13改 7x7模式
    # batch_t2_up = torch.reshape(hs_t2[:,1:-1,1:-1,:],(-1,channels)).to(device)

    batch_t1_up_no_reshape = torch.permute(batch_t1_up_no_reshape,(0,3,1,2)).to(device)
    batch_up_sampling = torch.nn.functional.interpolate(batch_t1_up_no_reshape, scale_factor=(3, 3), mode='bilinear').to(device)

    # 下路
    if channels == 224:
        # batch_t1_down = pan_t1.to(device).permute(0,3,1,2)
        batch_t2_down = pan_t2.to(device).permute(0,3,1,2)
    else:
        # batch_t1_down = pan_t1.unsqueeze(1).to(device)
        batch_t2_down = pan_t2.unsqueeze(1).to(device)

    # #对T1时刻图像进行空间维超分！！！！！！！！超超超
    # if need_Spatial_Super_Resolution == True:
    batch_t1_mid = nets_SSR(batch_t1_up_no_reshape,batch_up_sampling)
    if channels == 154:
        batch_t1_down = torch.mean(batch_t1_mid, dim=1).unsqueeze(1)
    else:
        batch_t1_down_R = torch.mean(batch_t1_mid[:,0:75,:,:], dim=1).unsqueeze(dim=1)
        batch_t1_down_G = torch.mean(batch_t1_mid[:,75:150,:,:], dim=1).unsqueeze(dim=1)
        batch_t1_down_B = torch.mean(batch_t1_mid[:,150:,:,:], dim=1).unsqueeze(dim=1)
        batch_t1_down = torch.cat((batch_t1_down_R,batch_t1_down_G,batch_t1_down_B),dim=1)

    # #对T2时刻图像进行光谱维超分！！！！！！！！超超超
    # if need_Spectral_Super_Resolution == True:
    batch_t2_mid = nets_SpeSR(batch_t2_down)
    batch_t2_up = torch.nn.functional.interpolate(batch_t2_mid,scale_factor=(1/3,1/3),mode='bilinear')
    batch_t2_up = torch.transpose(batch_t2_up, 1, 0)
    batch_t2_up = torch.reshape(batch_t2_up, (batch_t2_up.shape[0], -1))
    batch_t2_up = torch.transpose(batch_t2_up, 1, 0)

    # 下路CNN/上路MLP
    batch_t1_up_afterCNN = nets_upper[0](batch_t1_up)
    batch_t2_up_afterCNN = nets_upper[0](batch_t2_up)
    batch_t1_down_afterCNN = nets_lower[0](batch_t1_down, pool=True)
    batch_t2_down_afterCNN = nets_lower[0](batch_t2_down, pool=True)

    batch_t1_up_afterCNN = batch_t1_up_afterCNN.reshape((patch_num, -1, net_up_size1)) # 4.13改
    batch_t2_up_afterCNN = batch_t2_up_afterCNN.reshape((patch_num, -1, net_up_size1))
    # batch_t1_up_afterCNN = batch_t1_up_afterCNN.reshape((patch_num, patchsize_HSI * patchsize_HSI, net_up_size1))
    # batch_t2_up_afterCNN = batch_t2_up_afterCNN.reshape((patch_num, patchsize_HSI * patchsize_HSI, net_up_size1))

    # 每个patch构一张图，每个batch的图构成一个列表
    list_graph_up = []
    list_graph_down = []
    tensor_up_t1 = torch.zeros((patch_num, net_up_size1, patchsize_HSI * patchsize_HSI)).to(device)
    tensor_up_t2 = torch.zeros((patch_num, net_up_size1, patchsize_HSI * patchsize_HSI)).to(device)
    tensor_down_t1 = torch.zeros((patch_num, batch_t1_down_afterCNN.shape[3]*batch_t1_down_afterCNN.shape[3], net_down_size1)).to(device)
    tensor_down_t2 = torch.zeros((patch_num, batch_t1_down_afterCNN.shape[3]*batch_t1_down_afterCNN.shape[3], net_down_size1)).to(device)

    for patch in range(patch_num):
        graph_t1_up = build_graph_upper(batch_t1_up_afterCNN[patch, :, :], device)  # 每patch构一图
        graph_t2_up = build_graph_upper(batch_t2_up_afterCNN[patch, :, :], device)
        list_graph_up.append(graph_t1_up), list_graph_up.append(graph_t2_up)
        tensor_up_t1[patch, :, :] = graph_t1_up.x
        tensor_up_t2[patch, :, :] = graph_t2_up.x
        graph_t1_down = build_graph_lower(batch_t1_down_afterCNN[patch, :, :, :], device)  # 每patch构一图
        graph_t2_down = build_graph_lower(batch_t2_down_afterCNN[patch, :, :, :], device)
        list_graph_down.append(graph_t1_down), list_graph_down.append(graph_t2_down)
        tensor_down_t1[patch, :, :] = graph_t1_down.x
        tensor_down_t2[patch, :, :] = graph_t2_down.x

    batch_graph_up = Batch.from_data_list(list_graph_up).to(device)  # 每batch小图合成大图
    batch_graph_down = Batch.from_data_list(list_graph_down).to(device)  # 每batch小图合成大图

    # 中路CNN、加注意力、构图、GCN
    batch_t1_mid_afterCNN = nets_middle[0](batch_t1_mid, tensor_up_t1, tensor_down_t1, need_attention)
    batch_t2_mid_afterCNN = nets_middle[0](batch_t2_mid, tensor_up_t1, tensor_down_t2, need_attention)
    graph_t1_mid, A_t1, top_k = build_graph_middle(batch_t1_mid_afterCNN, device)  # 邻接矩阵在图匹配中还要用
    graph_t2_mid, A_t2, _ = build_graph_middle(batch_t2_mid_afterCNN, device)
    graph_t1_mid = graph_t1_mid.to(device)
    graph_t2_mid = graph_t2_mid.to(device)

    # 三路图匹配
    preds_mid, dif_mid = graph_matching.graph_matching_middle_new(graph_t1_mid, graph_t2_mid, nets_middle)
    preds_up, dif_up = graph_matching.graph_matching_UPorDOWN_new(batch_graph_up, nets_upper, patch_num)
    preds_down, dif_down = graph_matching.graph_matching_UPorDOWN_new(batch_graph_down, nets_lower, patch_num)


    if grad == 'no_grad':
        # 上下路差异向量
        dif = torch.cat((dif_up, dif_down,dif_mid), dim=1)

        # 三路预测向量
        if model == 1:
            preds_final = preds_mid
        elif model == 2:
            preds_final = preds_up
        elif model == 3:
            preds_final = preds_down
        else:
            preds_final = (preds_down + preds_up + preds_mid) / 3
        _, predicted_complex = torch.max(preds_final, 1)
        label_gt = list_gt.to(device)
        # 准确率
        correct = (predicted_complex == label_gt[:, 1]).sum().item()

        for num in range(label_gt.shape[0]):
            if (predicted_complex[num] == 0):
                result[arr[num]] = 0.5
            else:
                result[arr[num]] = 1
            if (label_gt[num, 1] == 0):
                gt[arr[num]] = 0.5
            else:
                gt[arr[num]] = 1

        batch_t1_mid = batch_t1_mid.reshape(patch_num,-1)
        batch_t1_mid = batch_t1_mid / torch.norm(batch_t1_mid,dim=0,keepdim=True)
        batch_t2_mid = batch_t2_mid.reshape(patch_num,-1)
        batch_t2_mid = batch_t2_mid / torch.norm(batch_t2_mid,dim=0,keepdim=True)
        distance = torch.sqrt(torch.sum((batch_t1_mid - batch_t2_mid) ** 2, dim=1))

        # 中路欧氏距离向量
        distances[(batch * batch_size):(batch * batch_size + patch_num)] = distance
        return dif,correct,result,gt,distances,preds_final


main()


