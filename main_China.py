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
import graph_matching_5_31 as graph_matching
from utils_try import adjust_learning_rate,build_graph_lower,build_graph_upper,build_graph_middle
import matplotlib.pyplot as plt
from Spatial_SR.common import default_conv
from Spatial_SR.SSPSR import SSPSR
from Spectral_SR.model_27 import SpeSRModel
from torch.utils.data import DataLoader,Subset
from dataset4_18 import AllDataset_no_mid as AllDataset
from HybridLoss import HybridLoss

name_data = 'China'
model = 0
percent = 20
percent2 = 0 # 5
num_renew_pseudo = 10 # 10
num_discard_orginal_label = 50
batch_size = 64
learning_rate = 1e-4
gpu = 'cuda:0'
alpha = 0.3
abstract = str(percent)+'%'+str(percent2)+'%'

if model == 0:
    pic_title = '67IRMAD_' + name_data + '_NewDecision_' + abstract
elif model == 1:
    pic_title = '67IRMAD_' + name_data + '_mid_' + abstract
elif model == 2:
    pic_title = '67IRMAD_' + name_data + '_up_' + abstract
elif model == 3:
    pic_title = '67IRMAD_' + name_data + '_down_' + abstract


# 全局变量
need_cut_data = False                                # 数据预处理

need_Spatial_Super_Resolution = True                 # 空间维超分
need_Spectral_Super_Resolution = True                # 光谱维超分

model_name = './Spatial_SR/pre_trained_model/pre_USA2.pth'  # 预训练空间超分模型 是这个吗？没用错吗
# model_name_Spectral = './Spectral_SR/pretrained_model/USA_pretrained.pth' # 预训练光谱超分模型 是这个吗？没用错吗
model_name_Spectral = './Spectral_SR/pretrained_model/USA_pretrained_150.pth' # 预训练光谱超分模型 是这个吗？没用错吗
pseudo_label_dir = './data/new_pseudo/IRMAD/result_China.mat'           # 5.4换伪标签1
pseudo_label_dir2 = './data/new_pseudo/PCA/result_China.mat'           # 5.4换伪标签2
need_save_model = True                              # 是否存模型
lower_pool_or_not = True                             # 下层用27x27还是pool到13x13,默认pool
load_origin_dir = './data/China.mat'
load_patch_dir = 'data/data_change/' + name_data



np.random.seed(100)
num_process_full_image = 10
epochs = 150
channels = 154
device = torch.device(gpu)
patchsize_HSI = 9
patchsize_PAN = 3 * patchsize_HSI

net_up_size1 = 64  # 上支路第一个MLP的输出尺寸 64 128
net_up_size2 = 128  # 上支路第一个MLP的隐层尺寸 128 256
net_down_size1 = 64 # 下支路第一个CNN的输出尺寸,GCN的输入尺寸 64
mid_hidden_size = 32 # 中路隐层尺寸

def main_cut():
    # 加载T1超分上来的高分高光谱图像（已切patch）和T2的PAN图像（整图）分别构成的patch列表
    if channels == 224:
        list_hs_t1,list_pan_t1,list_mid_t1 = data.create_data.create_HS_list_224(load_origin_dir, 't1', patchsize_HSI)
        list_hs_t2,list_pan_t2,list_mid_t2 = data.create_data.create_HS_list_224(load_origin_dir, 't2', patchsize_HSI)
        list_gt,high = data.create_data.create_gt(load_origin_dir)
        list_pseudo_label =  data.create_data.create_pseudo_label(pseudo_label_dir)
    elif name_data == 'USA':
        print('USA特供版数据预处理')
        list_hs_t1, list_pan_t1, list_mid_t1 = data.create_data_USA.create_HS_list(load_origin_dir, 't1', patchsize_HSI)
        list_hs_t2, list_pan_t2, list_mid_t2 = data.create_data_USA.create_HS_list(load_origin_dir, 't2', patchsize_HSI)
        list_gt, high = data.create_data_USA.create_gt(load_origin_dir)
        list_pseudo_label = data.create_data_USA.create_pseudo_label(pseudo_label_dir)
    else:
        list_hs_t1,list_pan_t1,list_mid_t1 = data.create_data.create_HS_list(load_origin_dir, 't1', patchsize_HSI)
        list_hs_t2,list_pan_t2,list_mid_t2 = data.create_data.create_HS_list(load_origin_dir, 't2', patchsize_HSI)
        list_gt,high = data.create_data.create_gt(load_origin_dir)
        list_pseudo_label = data.create_data.create_pseudo_label(pseudo_label_dir)

    arr = np.arange(len(list_gt))

    # 打印一下伪标签和真值
    list_pseudo_label_np = np.array(list_pseudo_label)
    list_gt2 = np.array(list_gt)
    if channels == 224:
        num_right = np.count_nonzero((list_gt2 == 1) & (list_pseudo_label_np == 1))
        num_right = num_right + np.count_nonzero((list_gt2 == 2) & (list_pseudo_label_np == 0))
        num_sum = np.count_nonzero(list_gt2==2)
        num_sum = num_sum + np.count_nonzero(list_gt2==1)
        correct = num_right/num_sum
    else:
        correct = np.sum(np.equal(list_gt, list_pseudo_label_np))
    list_pseudo_label_np = np.reshape(list_pseudo_label_np, (high, int(len(list_gt) / high)))
    list_gt2 = np.reshape(list_gt2, (high, int(len(list_gt) / high)))
    fig, axs = plt.subplots(1, 2)
    if channels == 224:
        fig.suptitle('pseudo & gt of ' + name_data + ',num of same:' + str(num_right) + '/' + str(num_sum))
    else:
        fig.suptitle('pseudo & gt of ' + name_data +',num of same:' + str(correct))
    axs[0].imshow(list_gt2, cmap='gray')
    axs[0].set_title('gt')
    axs[1].imshow(list_pseudo_label_np, cmap='gray')
    axs[1].set_title('pseudo_label')
    plt.show()

    list_gt = torch.tensor(list_gt)
    if channels==154:
        list_gt = torch.nn.functional.one_hot(list_gt, 2).type(torch.FloatTensor)
    elif channels == 224:
        list_gt = torch.nn.functional.one_hot(list_gt, 3).type(torch.FloatTensor)
        list_gt[:,[0,1,2]] = list_gt[:,[2,1,0]]
        # 改变后: 0不变 1变 2背景
    else:
        raise Exception('unknown number of channels:',channels,'!!!')

    print('开始切patch')
    print('patch save_dit:./data/data_change/'+name_data+'/')
    time1 = time.time()
    time_all = time.time()
    ii = 0
    for i in range(len(list_gt)):
        if (list_gt.shape[1]==2) or (list_gt.numpy()[i,2] != 1):
            mdict = {'hs_t1':list_hs_t1[i],'hs_t2':list_hs_t2[i],'pan_t1':list_pan_t1[i],'pan_t2':list_pan_t2[i],
                     'gt':list_gt.numpy()[i,:],'arr':arr[i],'pseudo_label':list_pseudo_label[i]}
            sio.savemat('./data/data_change/'+name_data+'/'+str(ii)+'.mat',mdict=mdict)
            ii = ii + 1
            if (ii+1)%1000==0:
                print(ii+1,'patch已切,1000个patch用时',time.time()-time1)
                time1 = time.time()
    print('已切完,总用时',time.time()-time_all)

def main():
    global alpha
    print(pic_title)
    mdict_all = {}
    loca = time.strftime('%Y-%m-%d')
    print('======> start loading the image!')
    # 将patch读入dataloader
    all_dataset = AllDataset(load_patch_dir)
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    arr_all = []
    for batch, (hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label, _) in enumerate(
            all_dataloader):
        for i in arr:
            arr_all.append(i.item())

    print('loading dataset,dir = .', load_patch_dir)
    if name_data == 'USA':
        gt, height = data.create_data_USA.create_gt(load_origin_dir)
    else:
        gt, height = data.create_data.create_gt(load_origin_dir)
    width = int(len(gt) / height)
    num_data = len(all_dataset)
    print('数据点总数:{}'.format(num_data))

    # 每轮新增伪标签数
    num_add_label_change = int(num_data/100)
    num_add_label_unchange = int(num_data/100)

    # 读入伪标签
    list_pseudo_label,img_pseudo_label = data.create_data.create_pseudo_label(pseudo_label_dir,need_numpy=True)
    list_pseudo_label2 = data.create_data.create_pseudo_label(pseudo_label_dir2)
    # 杂糅两个伪标签
    pseudo_label_same = []
    pseudo_label_dif = []

    if channels == 154:
        for i in range(len(list_pseudo_label)):
            if list_pseudo_label[i] == list_pseudo_label2[i]:
                pseudo_label_same.append(i)
            else:
                pseudo_label_dif.append(i)
    elif channels == 224:
        datas = sio.loadmat(load_origin_dir)
        Binary = np.array(datas['Binary'][...], dtype=np.float32)
        Binary = np.pad(Binary, ((0, 0), (0, 1)), 'symmetric')
        arr_background = []
        for i in range(int(Binary.shape[0] / 3)):
            for j in range(int(Binary.shape[1] / 3)):
                if Binary[i * 3 + 1, j * 3 + 1] == 0:
                    arr_background.append(1)
                else:
                    arr_background.append(0)
        del Binary
        k = 0
        for i in range(len(list_pseudo_label)):
            # if (list_pseudo_label[i] != list_pseudo_label2[i]) & (arr_background[i]==0):
            if (list_pseudo_label[i] == list_pseudo_label2[i]) & (arr_background[i]==0):
                pseudo_label_same.append(k)
                k = k + 1
            elif arr_background[i]==0:
                pseudo_label_dif.append(k)
                k = k + 1
        del arr_background

    del list_pseudo_label, list_pseudo_label2
    img_pseudo_label = np.pad(img_pseudo_label, ((4, 4), (4, 4)), 'symmetric')

    print('两组伪标签相同的点数：',len(pseudo_label_same),'两组伪标签不同的点数：',len(pseudo_label_dif))
    pseudo_label_same = sample(pseudo_label_same,int(len(pseudo_label_same)*(percent/100)))
    subset_a = Subset(all_dataset, pseudo_label_same)
    print('相同伪标签选{}%'.format(percent))
    if percent2 == 0:
        subset = Subset(dataset = all_dataset,indices=pseudo_label_same)
        del subset_a
    else:
        pseudo_label_dif = sample(pseudo_label_dif, int(len(pseudo_label_dif) *(percent2/100)))
        subset_b = Subset(all_dataset, pseudo_label_dif)
        print('不同伪标签选{}%，以准确性高的伪标签为准:'.format(percent2))
        subset = Subset(dataset=all_dataset, indices=pseudo_label_same + pseudo_label_dif)
        print('训练所选取伪标签点总数：{}'.format(len(subset_a) + len(subset_b)))
        del subset_a, subset_b
    sub_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 读入T1时刻低分
    datas = sio.loadmat(load_origin_dir)
    img1 = np.array(datas['T1'][...], dtype=np.float32)
    img1 = data.create_data.normalize_data(img1)
    temp = img1[:, :, 0]  # 420x140
    if  name_data == 'USA':
        temp2 = np.pad(temp, ((0, 1), (0, 2)), 'symmetric')
    else:
        temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
    [h_pad, w_pad] = temp2.shape
    img_pad = np.empty((h_pad, w_pad, channels), dtype='float32')
    for i in range(channels):
        temp = img1[:, :, i]
        if name_data == 'USA':
            temp2 = np.pad(temp, ((0, 1), (0, 2)), 'symmetric')
        else:
            temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
        img_pad[:, :, i] = temp2
    img_pad = torch.from_numpy(img_pad).to(device)
    img_pad = torch.permute(img_pad,(2,0,1))
    img_pad = torch.unsqueeze(img_pad,dim=0)
    img_lr = torch.nn.functional.interpolate(img_pad, scale_factor=(1/3, 1/3),mode='bilinear')
    img_lr_upsampling = torch.nn.functional.interpolate(img_lr, scale_factor=(3, 3),mode='bilinear')
    print('T1时刻低分辨率图的大小：',img_lr.shape)

    # 读入T2时刻PAN
    img2 = np.array(datas['T2'][...], dtype=np.float32)
    img2 = data.create_data.normalize_data(img2)
    temp = img2[:, :, 0]  # 420x140
    if name_data == 'USA':
        temp2 = np.pad(temp, ((0, 1), (0, 2)), 'symmetric')
    else:
        temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
    [h_pad, w_pad] = temp2.shape
    img_pad = np.empty((h_pad, w_pad, channels), dtype='float32')
    for i in range(channels):
        temp = img2[:, :, i]
        if name_data == 'USA':
            temp2 = np.pad(temp, ((0, 1), (0, 2)), 'symmetric')
        else:
            temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
        img_pad[:, :, i] = temp2
    if channels == 154:
        img_pan = np.mean(img_pad[:,:,:27], axis=2)
        img_pan = data.create_data.mat2gray(img_pan)
        img_pan = torch.from_numpy(img_pan).to(device)
    else:
        # img_R = np.mean(img_pad[:, :, 0:75], axis=2)
        # img_R = data.create_data.mat2gray(img_R)
        # img_G = np.mean(img_pad[:, :, 75:150], axis=2)
        # img_G = data.create_data.mat2gray(img_G)
        # img_B = np.mean(img_pad[:, :, 150:224], axis=2)
        # img_B = data.create_data.mat2gray(img_B)
        # img_RGB = np.zeros((img_R.shape[0],img_R.shape[1],3))
        # img_RGB[:,:,0] = img_R
        # img_RGB[:,:,1] = img_G
        # img_RGB[:,:,2] = img_B
        # img_RGB = torch.from_numpy(img_RGB).to(device).to(torch.float32).permute(2,0,1).unsqueeze(dim=0)
        img_RGB = np.zeros((img_pad.shape[0], img_pad.shape[1], 3))
        img_RGB[:, :, 0] = img_pad[:, :, 10]
        img_RGB[:, :, 1] = img_pad[:, :, 20]
        img_RGB[:, :, 2] = img_pad[:, :, 35]
        img_RGB = torch.from_numpy(img_RGB).to(device)
        print('T2时刻RGB图的大小：', img_RGB.shape)

    # 图匹配网
    nets_lower = [nets.CNN(net_size1=net_up_size1),nets.GCN(net_size1=net_up_size1),nets.message_net(),nets.MLP(),nets.MLP2(),nets.MLP3()]
    nets_upper = [nets.MLP0(channels=channels, net_size1=net_up_size1,net_size2=net_up_size2),
                  nets.GCNup(patchsize=patchsize_HSI), nets.message_netup(), nets.MLPup(), nets.MLP2up(), nets.MLP3()]
    nets_middle = [
        nets.CNNmid(channel=channels, patchsize=patchsize_HSI, down_size=net_down_size1, out_size=net_up_size1,
                        hidden_size=mid_hidden_size),
        nets.GCNmid(), nets.linear_difference(), nets.linear_classification(),nets.linear_attention()]  # 阿巴阿巴阿巴巴


    # 额外网，用来加入额外标签
    NetsExtra = nets.NetsExtra().to(device)

    # 空间超分网
    nets_SSPSR = SSPSR(n_subs=8, n_ovls=1, n_colors=channels, n_blocks=3, n_feats=256,
            n_scale=3, res_scale=0.1, use_share=True, conv=default_conv)
    state_dict = torch.load(model_name)
    nets_SSPSR.load_state_dict(state_dict)
    if need_Spatial_Super_Resolution:
        print('use pre_trained SSPSR model name:', model_name)
    nets_SSPSR.to(device)

    # 光谱超分网
    nets_SpeSR = SpeSRModel(input_features=1, output_features=channels)
    state_dict2 = torch.load(model_name_Spectral)
    nets_SpeSR.load_state_dict(state_dict2)

    if need_Spectral_Super_Resolution:
        print('use pre_trained SpeSR model name:', model_name_Spectral)
    nets_SpeSR.to(device)

    # 参数更新
    parameters = []
    parameters_noSR = []
    # 三个图匹配网
    for net in nets_upper:
        net.to(device)
        parameters = parameters + list(net.parameters())
        parameters_noSR = parameters_noSR + list(net.parameters())
    for net in nets_lower:
        net.to(device)
        parameters = parameters + list(net.parameters())
        parameters_noSR = parameters_noSR + list(net.parameters())
    for net in nets_middle:
        net.to(device)
        parameters = parameters + list(net.parameters())
        parameters_noSR = parameters_noSR + list(net.parameters())
    # 两个超分网
    if need_Spatial_Super_Resolution:
        parameters = parameters + list(nets_SSPSR.parameters())
    if need_Spectral_Super_Resolution:
        parameters = parameters + list(nets_SpeSR.parameters())

    # 损失与优化器
    loss_classification = torch.nn.CrossEntropyLoss().to(device)
    # loss_classification = TruncatedLoss(trainset_size=len(all_dataset)).to(device)
    # 4.18 15:01换用新损失    # 4.24换回老损失

    loss_MSE = torch.nn.MSELoss().to(device)
    loss_L1 = torch.nn.L1Loss().to(device)
    # loss_L1 = HybridLoss(spatial_tv=True, spectral_tv=True).to(device)
    optimizer = Adam(parameters , lr=learning_rate, weight_decay=0)
    optimizer_noSR = Adam(parameters_noSR, lr=learning_rate, weight_decay=0)
    optimizer2 = Adam(NetsExtra.parameters(),lr=learning_rate,weight_decay=0)

    accuracy_max = 0

    # 对于每次迭代
    print('============> start training!')
    for e in range(epochs):
        time_begin = time.time()
        print('============> <{}/{}>epoch is beginning'.format(e + 1, epochs))
        adjust_learning_rate(learning_rate, optimizer, e + 1)
        if e == 0:
            arr_append1 = None
            arr_append0 = None

        if ((e + 1) % num_process_full_image == 0) or e == 0:
            process_full_image = True
        else:
            process_full_image = False
        if ((e + 1) >= num_renew_pseudo)&(e < 100):
            renew_pseudo = True
        else:
            renew_pseudo = False

        # 超分结果打印
        # '''
        if process_full_image:
            with torch.no_grad():
                nets_SSPSR.eval()
                nets_SpeSR.eval()
                img_t1_after_sr = nets_SSPSR(img_lr, img_lr_upsampling)
                img_t1_after_sr = torch.squeeze(img_t1_after_sr)

                if e==0:
                    if name_data == 'USA':
                        img_pan = img_pan.cpu().numpy()
                        img_pan = np.pad(img_pan, ((7, 8), (0, 0)), 'symmetric')
                        num_patches_h = img_pan.shape[0] // (patchsize_PAN - 6)
                        num_patches_w = img_pan.shape[1] // (patchsize_PAN - 6)
                    elif name_data == 'China':
                        img_pan = img_pan.cpu().numpy()
                        img_pan = np.pad(img_pan, ((0, 0), (3, 3)), 'symmetric')
                        num_patches_h = img_pan.shape[0] // (patchsize_PAN - 6)
                        num_patches_w = img_pan.shape[1] // (patchsize_PAN - 6)
                    elif name_data == 'Bar':
                        img_RGB = img_RGB.cpu().numpy()
                        img_RGB = np.pad(img_RGB, ((0, 3), (8, 8), (0, 0)), 'symmetric')
                        num_patches_h = img_RGB.shape[0] // (patchsize_PAN - 6)
                        num_patches_w = img_RGB.shape[1] // (patchsize_PAN - 6)
                    else:
                        img_RGB = img_RGB.cpu().numpy()
                        img_RGB = np.pad(img_RGB, ((3, 6), (1, 2), (0, 0)), 'symmetric')
                        num_patches_h = img_RGB.shape[0] // (patchsize_PAN - 6)
                        num_patches_w = img_RGB.shape[1] // (patchsize_PAN - 6)

                # 初始化高光谱和全色 patch 列表
                if channels == 154:
                    img_t2_after_sr = torch.zeros((channels, img_pan.shape[0], img_pan.shape[1])).to(device)
                    for i in range(num_patches_h):
                        for j in range(num_patches_w):
                            # 计算 patch 范围
                            h_start, h_end = i * 21, (i + 1) * 21
                            w_start, w_end = j * 21, (j + 1) * 21
                            # 切分 patch
                            temp = img_pan[h_start:h_end, w_start:w_end]
                            temp = np.pad(temp, ((3, 3), (3, 3),), 'symmetric')
                            temp = torch.from_numpy(temp)
                            pan_patch = temp.unsqueeze(dim=0).unsqueeze(dim=0).to(device).float()
                            # print(pan_patch.shape) # 1x3x27x27
                            pan_patch = nets_SpeSR(pan_patch)[:,:, 3:-3, 3:-3]
                            # print(pan_patch.shape)  # 1x224x21x21
                            img_t2_after_sr[:, h_start:h_end, w_start:w_end] = pan_patch

                elif channels == 224:
                    img_t2_after_sr = torch.zeros((channels, img_RGB.shape[0], img_RGB.shape[1])).to(device)
                    for i in range(num_patches_h):
                        for j in range(num_patches_w):
                            # 计算 patch 范围
                            h_start, h_end = i * 21, (i + 1) * 21
                            w_start, w_end = j * 21, (j + 1) * 21
                            # 切分 patch
                            temp = img_RGB[h_start:h_end, w_start:w_end,:]
                            temp = np.pad(temp, ((3, 3), (3, 3), (0, 0)), 'symmetric')
                            temp = torch.from_numpy(temp)
                            pan_patch = temp.unsqueeze(dim=0).to(device).permute(0, 3, 1, 2).float()
                            # print(pan_patch.shape) # 1x3x27x27
                            pan_patch = nets_SpeSR(pan_patch)[:, :, 3:-3, 3:-3]
                            # print(pan_patch.shape)  # 1x224x21x21
                            img_t2_after_sr[:, h_start:h_end, w_start:w_end] = pan_patch

                if True:
                    # 确认拼接图像尺寸与原图像尺寸一致
                    if name_data == 'USA':
                        img_t2_after_sr = img_t2_after_sr[:, 7:-8, :]
                    elif name_data == 'China':
                        img_t2_after_sr = img_t2_after_sr[:, :, 3:-3]
                    elif name_data == 'Bar':
                        img_t2_after_sr = img_t2_after_sr[:, :-3, 8:-8]
                    else:
                        img_t2_after_sr = img_t2_after_sr[:, 3:-6, 1:-2]

                    image_t1_sr = img_t1_after_sr[(140, 80, 20), :, :].detach().cpu().numpy()
                    image_t1_sr = image_t1_sr.transpose(1, 2, 0)
                    image_t2_sr = img_t2_after_sr[(140, 80, 20), :, :].detach().cpu().numpy()
                    image_t2_sr = image_t2_sr.transpose(1, 2, 0)
                    image_t1_pan = np.mean(image_t1_sr, axis=2)
                    image_t1_pan = data.create_data.mat2gray(image_t1_pan)
                    image_t2_pan = np.mean(image_t2_sr, axis=2)
                    image_t2_pan = data.create_data.mat2gray(image_t2_pan)

                    fig, axs = plt.subplots(1, 4)
                    fig.suptitle('epoch' + str(e + 1) + ', '+pic_title+',show the SR:')
                    axs[0].imshow(image_t1_sr)
                    axs[0].set_title('t1')
                    axs[2].imshow(image_t2_sr)
                    axs[2].set_title('t2')
                    axs[1].imshow(image_t1_pan, cmap='gray')
                    axs[1].set_title('t1_pan')
                    axs[3].imshow(image_t2_pan, cmap='gray')
                    axs[3].set_title('t2_pan')
                    plt.show()
                    # img_t1_after_sr = img_t1_after_sr.detach().cpu().numpy().transpose(1,2,0)
                    # img_t2_after_sr = img_t2_after_sr.detach().cpu().numpy().transpose(1,2,0)
                    # sio.savemat('./result/SR_'+name_data+'-'+str(e+1)+'-'+loca+'.mat',mdict={'t1':img_t1_after_sr,'t2':img_t2_after_sr} )
        # '''
        # 本次迭代内变量
        loss_sum = 0  # 本次迭代总loss
        correct_sum = 0  # 本次迭代总正确分类数
        dif_all = torch.zeros((num_data, 128 * 2))  # 上下路差异向量，用于聚类
        distances = torch.zeros(num_data)  # 中路batch间欧氏距离
        result = np.zeros(height * width)  # 分类结果
        gt_for_show = np.zeros(height * width)  # 用于展示的标签
        gt = np.zeros(height * width)  # 真实标签，用于计算正确率
        # pseudo_labels = np.zeros(num_data)  # 传统算法全图伪标签
        have_label_or_not = torch.zeros(num_data)
        arrs = np.zeros(height * width).astype(int)
        label_for_show = np.zeros((height * width,3)).astype(int)
        preds_all = torch.zeros(num_data)

        # 全图过程：每十轮一次，跑验证，测结果，更新伪标签
        if process_full_image:
            if (e != 0) & (alpha > 0.1):
                alpha = alpha * 0.8
            if alpha < 0.1:
                alpha = 0.1
            print('e', (e + 1), '_alpha:', alpha)
            for net in nets_upper:
                net.eval()
            for net in nets_lower:
                net.eval()
            for net in nets_middle:
                net.eval()
            nets_SSPSR.eval()
            nets_SpeSR.eval()

            with torch.no_grad():
                time_no_loss_begin = time.time()

                for batch, (hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label, _) in enumerate(
                        all_dataloader):
                    dif, correct, result, gt_for_show, distances, preds_2 = batch_process(
                        'no_grad', batch, hs_t1, hs_t2, pan_t1, pan_t2,
                        nets_upper, nets_lower, nets_middle, nets_SSPSR, nets_SpeSR,
                        labels=pseudo_label, list_gt=list_gt, arr=arr, result=result, gt=gt_for_show,
                        distances=distances, e=e)

                    # pseudo_labels[batch * batch_size:batch * batch_size + len(hs_t1)] = pseudo_label
                    gt[batch * batch_size:batch * batch_size + len(hs_t1)] = list_gt[:, 1]
                    arrs[batch * batch_size:batch * batch_size + len(hs_t1)] = arr
                    correct_sum = correct_sum + correct
                    preds_all[batch * batch_size:batch * batch_size + len(hs_t1)] = preds_2[:, 0] - preds_2[:, 1]
                    dif_all[batch * batch_size:batch * batch_size + len(hs_t1), :] = dif
                k = int(len(preds_all) * 0.2)
                topk_values, topk_indices = torch.topk(preds_all, k)
                bottomk_values, bottomk_indices = torch.topk(preds_all, k, largest=False)

                print('batch num for no_loss iteration:', batch + 1, '.times:', time.time() - time_no_loss_begin)

                # 构造 have_label_or_not ，如果一个样本在sub_dataloader里，相应位置置1
                have_label_or_not[subset.indices] = 1

            if renew_pseudo:
                time_extra = time.time()
                NetsExtra.train()
                have_label_or_not = have_label_or_not.to(device)
                dif_all = dif_all.to(device)
                for j in range(10):
                    for i in range(math.floor(num_data/batch_size)):
                        a = dif_all[i*batch_size:(i+1)*batch_size,:]
                        a = NetsExtra(a).squeeze()
                        b = have_label_or_not[i*batch_size:(i+1)*batch_size]
                        # loss = loss_MSE(a,b)+loss_L1(a,b)
                        loss = loss_classification(a,b)
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer2.step()
                NetsExtra.eval()
                dif_all = NetsExtra(dif_all)
                # median = torch.median(dif_all)
                # indices_media = torch.nonzero(torch.lt(dif_all,median))
                dif_sorted_indices = torch.argsort(dif_all).detach().cpu().tolist()
                selected_indices = random.sample(range(len(dif_sorted_indices)),int(len(dif_sorted_indices)/10))
                dif_sorted_indices2 = [dif_sorted_indices[i] for i in sorted(selected_indices)]
                dif_sorted_indices = dif_sorted_indices2
                del dif_sorted_indices2,selected_indices
                top = 0                 # top是0
                bottom = 0              # bottom是1
                top_true = 0
                bottom_true = 0
                top_right = 0
                bottom_right = 0
                arr_append0_pre = []
                arr_append1_pre = []

                for i in dif_sorted_indices:
                    if i in topk_indices:
                        top = top + 1
                        if have_label_or_not[i] == 0 :
                            arr_append0_pre.append(i)
                            top_true = top_true + 1
                            if gt[i] == 0:
                                top_right = top_right + 1
                    if top_true > num_add_label_unchange - 1:
                        break

                for i in dif_sorted_indices:
                    if i in bottomk_indices:
                        bottom = bottom + 1
                        if have_label_or_not[i] == 0 :
                            arr_append1_pre.append(i)
                            bottom_true = bottom_true + 1
                            if gt[i] == 1:
                                bottom_right = bottom_right + 1
                    if bottom_true > num_add_label_change - 1:
                        break

                # a = np.zeros((height * width), dtype=int)
                # b = np.zeros((height * width), dtype=int)
                # c = np.zeros((height * width), dtype=int)
                # for i in arr_append0_pre:
                #     a[arr_all[i]] = 225
                # for i in arr_append1_pre:
                #     b[arr_all[i]] = 255
                # have_label_or_not = have_label_or_not.detach().cpu().numpy().astype(int)
                # for i in range(num_data):
                #     if have_label_or_not[i] == 1:
                #         c[arr_all[i]] = 255
                # a = a.reshape((height, width))
                # b = b.reshape((height, width))
                # c = c.reshape((height, width))
                #
                # fig, axs = plt.subplots(1, 3)
                # plt.suptitle(str(e + 1) + '_new_label')
                # axs[0].imshow(a, cmap='gray')
                # axs[0].set_title('new_0')
                # axs[1].imshow(b, cmap='gray')
                # axs[1].set_title('new_1')
                # axs[2].imshow(c, cmap='gray')
                # axs[2].set_title('old_label')
                # plt.show()
                # del a, b, c, have_label_or_not
                print('<<<<<', e + 1, 'epoch,top:', top, 'top_true:', top_true, 'top_right:', top_right, ',bottom:',
                      bottom, ',bottom_true:', bottom_true, ',bottom_right:', bottom_right, '>>>>>')

                if arr_append1 == None:
                    arr_append1 = []
                    arr_append0 = []
                arr_append0 = arr_append0 + arr_append0_pre
                arr_append1 = arr_append1 + arr_append1_pre
                print('重构前的训练集总长度为：', len(subset))
                if e>=num_discard_orginal_label:
                    subset = Subset(dataset=all_dataset, indices=arr_append0 + arr_append1)
                else:
                    subset = Subset(dataset=all_dataset, indices=subset.indices + arr_append0_pre + arr_append1_pre)
                sub_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
                print('重构后的训练集总长度为：',len(subset))

                print('train the extra net need time:', time.time() - time_extra)

            gtgt = np.zeros((height * width))
            for batch2, (hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label, index) in enumerate(
                    sub_dataloader):
                for i in range(hs_t1.shape[0]):
                    if list_gt[i, 1] == 1:
                        gtgt[arr[i]] = 1
                    else:
                        gtgt[arr[i]] = 2

        # time_with_loss_begin = time.time()
        for net in nets_upper:
            net.train()
        for net in nets_lower:
            net.train()
        for net in nets_middle:
            net.train()
        nets_SSPSR.train()
        nets_SpeSR.train()

        sum_sum = [0, 0, 0, 0]

        label_true = 0
        label_false = 0
        label_true_new = 0
        label_false_new = 0

        for batch2, (hs_t1, hs_t2, pan_t1, pan_t2, list_gt, arr, pseudo_label, index) in enumerate(sub_dataloader):
            if process_full_image:
                for i in range(hs_t1.shape[0]):
                    if list_gt[i, 1] == pseudo_label[i]:
                        if not renew_pseudo:
                            label_for_show[arr[i], 1] = 255
                        label_true = label_true + 1
                    else:
                        if not renew_pseudo:
                            label_for_show[arr[i], 0] = 255
                        label_false = label_false + 1
                if renew_pseudo:
                    for i in range(hs_t1.shape[0]):
                        if index[i] in arr_append0:
                            if gtgt[arr[i]] == 2:
                                label_for_show[arr[i], 1] = 255
                                label_true_new = label_true_new + 1
                            else:
                                label_for_show[arr[i], 0] = 255
                                label_false_new = label_false_new + 1
                    for i in range(hs_t1.shape[0]):
                        if index[i] in arr_append1:
                            if gtgt[arr[i]] == 1:
                                label_for_show[arr[i], 1] = 255
                                label_for_show[arr[i], 2] = 255
                                label_true_new = label_true_new + 1
                            else:
                                label_for_show[arr[i], 0] = 255
                                label_for_show[arr[i], 2] = 255
                                label_false_new = label_false_new + 1

            loss,Sum = batch_process('with_grad', batch2, hs_t1, hs_t2, pan_t1, pan_t2, nets_upper, nets_lower, nets_middle,
                                     nets_SSPSR, nets_SpeSR,
                                     labels=pseudo_label, list_gt=list_gt, loss_classification=loss_classification, loss_MSE=loss_MSE, loss_L1=loss_L1,
                                     arr=arr, optimizer=optimizer, e=e, arr_append0=arr_append0, arr_append1=arr_append1, index=index, ipl1 = img_pseudo_label)
            loss_sum = loss_sum + loss
            sum_sum[0] = sum_sum[0] + Sum[0]
            sum_sum[1] = sum_sum[1] + Sum[1]
            sum_sum[2] = sum_sum[2] + Sum[2]
            sum_sum[3] = sum_sum[3] + Sum[3]
        # print('batch num for with_loss iteration:', batch2 + 1, '.times:', time.time() - time_with_loss_begin)

        if process_full_image:
            print('新增正确标签{},错误标签{}'.format(label_true_new, label_false_new))
            print('[总计:{},全部一致:{},旧标签错误:{},新标签错误:{},新伪标签都错误:{}]'.format(
                sum_sum[0] + sum_sum[1] + sum_sum[2] + sum_sum[3], sum_sum[0], sum_sum[1], sum_sum[2], sum_sum[3]))
            print('总共正确标签{},错误标签{}'.format(label_true, label_false))
            # 打印结果图
            result = np.reshape(result, (height, width))

            if e > 30:
                img_pseudo_label = np.pad(result, ((4, 4), (4, 4)), 'symmetric')
                img_pseudo_label[img_pseudo_label==0.5] = 0

            gt_for_show = np.reshape(gt_for_show, (height, width))
            label_for_show = np.reshape(label_for_show,(height,width,3))
            accuracy = correct_sum / num_data

            fig, axs = plt.subplots(1, 3)
            fig.suptitle(str(e + 1) + ' ' + pic_title + '\n' + 'accuracy:' + str(correct_sum / num_data))
            axs[0].imshow(gt_for_show, cmap='gray')
            axs[0].set_title(str(e + 1) + ' gt_for_show')
            axs[1].imshow(result, cmap='gray')
            axs[1].set_title(str(e + 1) + ' result')
            axs[2].imshow(label_for_show)
            axs[2].set_title('label' + str(label_true) +'/' + str(label_true+label_false))
            plt.show()

            mdict_all['e'+str(e+1)] = result
            if (e!=0)&(accuracy>0.9):
                if accuracy > accuracy_max:
                    accuracy_max = accuracy
                    e_max = e+1
                print(accuracy_max)
                accu = f'{accuracy:.3f}'

                if (need_save_model == True):
                    dir_save = './save_model/' + name_data + '/' + pic_title + '_' + accu
                    if not os.path.exists(dir_save):
                        os.makedirs(dir_save)
                        os.makedirs(dir_save + '/up')
                        os.makedirs(dir_save + '/low')
                        os.makedirs(dir_save + '/middle')

                    k = 0
                    for net in nets_upper:
                        torch.save(net.state_dict(), dir_save + '/up/' + str(k) + '.pth')
                        k = k + 1
                    k = 0
                    for net in nets_lower:
                        torch.save(net.state_dict(), dir_save + '/low/' + str(k) + '.pth')
                        k = k + 1
                    k = 0
                    for net in nets_middle:
                        torch.save(net.state_dict(), dir_save + '/middle/' + str(k) + '.pth')
                        k = k + 1
                    torch.save(nets_SpeSR.state_dict(), dir_save + '/SpeSR' + '.pth')
                    torch.save(nets_SSPSR.state_dict(), dir_save + '/SSPSR' + '.pth')

                '''
                sio.savemat('./result/'+pic_title+str(batch_size)+'-'+str(e+1)+'-'+accu+'-'+loca+'.mat', mdict = {'result':result})
                '''
            print(
                '<{}/{}>epoch has done,Time for this epoch: {:.2f} seconds,avg loss in this epoch：{:.6f},avg correct in this epoch：{:.6f}'
                .format(e + 1, epochs, time.time() - time_begin, loss_sum / len(sub_dataloader),accuracy))
        else:
            print(
                '<{}/{}>epoch has done,Time for this epoch: {:.2f} seconds,avg loss in this epoch：{:.6f}'
                .format(e + 1, epochs, time.time() - time_begin, loss_sum / len(sub_dataloader)))

    # 保存模型
    '''
    if (need_save_model == True):
        dir_save = './save_model/' + name_data + '/' + pic_title
        os.makedirs(dir_save)
        os.makedirs(dir_save + '/up')
        os.makedirs(dir_save + '/low')
        os.makedirs(dir_save + '/middle')
        k = 0
        for net in nets_upper:
            torch.save(net.state_dict(), dir_save + '/up/' + str(k) + '.pth')
            k = k + 1
        k = 0
        for net in nets_lower:
            torch.save(net.state_dict(), dir_save + '/low/' + str(k) + '.pth')
            k = k + 1
        k = 0
        for net in nets_lower:
            torch.save(net.state_dict(), dir_save + '/middle/' + str(k) + '.pth')
            k = k + 1
        torch.save(nets_SpeSR.state_dict(), dir_save + '/SpeSR' + '.pth')
        torch.save(nets_SSR.state_dict(), dir_save + '/SSPSR' + '.pth')
    '''

    print(accuracy_max)
    accu = f'{accuracy_max:.3f}'
    sio.savemat('./result/result_China/' + pic_title + '-' + str(batch_size) + '-' + str(learning_rate) +'-' +  str(e_max) +'-' + accu + '-' + str(num_renew_pseudo) + '-' + str(num_discard_orginal_label) + '-' + loca + '.mat', mdict = mdict_all)


def batch_process(grad,batch,hs_t1,hs_t2,pan_t1,pan_t2,nets_upper,nets_lower,nets_middle,nets_SSR,nets_SpeSR,
                  labels=None,loss_classification=None,loss_MSE=None,loss_L1=None,optimizer=None,list_gt=None,arr=None,result=None,
                  gt=None,distances=None,e=None,arr_append0=None,arr_append1=None,index=None,ipl1=None):

    global alpha
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
    batch_t1_down_afterCNN = nets_lower[0](batch_t1_down, lower_pool_or_not)
    batch_t2_down_afterCNN = nets_lower[0](batch_t2_down, lower_pool_or_not)

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
    batch_t1_mid_afterCNN = nets_middle[0](batch_t1_mid, tensor_up_t1, tensor_down_t1, need_attention=True)
    batch_t2_mid_afterCNN = nets_middle[0](batch_t2_mid, tensor_up_t1, tensor_down_t2, need_attention=True)
    graph_t1_mid, A_t1, top_k = build_graph_middle(batch_t1_mid_afterCNN, device)  # 邻接矩阵在图匹配中还要用
    graph_t2_mid, A_t2, _ = build_graph_middle(batch_t2_mid_afterCNN, device)
    graph_t1_mid = graph_t1_mid.to(device)
    graph_t2_mid = graph_t2_mid.to(device)

    # 三路图匹配
    preds_mid, dif_mid, s_diag = graph_matching.graph_matching_middle_new(graph_t1_mid, graph_t2_mid, nets_middle)
    preds_up, dif_up = graph_matching.graph_matching_UPorDOWN_new(batch_graph_up, nets_upper, patch_num, s_diag)
    preds_down, dif_down = graph_matching.graph_matching_UPorDOWN_new(batch_graph_down, nets_lower, patch_num, s_diag)

    if(grad == 'with_grad'):
        # 分类损失
        label = torch.from_numpy(np.eye(2)[labels]).to(device)
        if model == 1:
            loss = loss_classification(preds_mid, label)
        elif model == 2:
            loss = loss_classification(preds_up, label)
        elif model == 3:
            loss = loss_classification(preds_down, label)
        else:
            loss1 = loss_classification(preds_mid, label)
            loss2 = loss_classification(preds_up, label)
            loss3 = loss_classification(preds_down, label)
            loss = loss1 + loss2 + loss3

        # 图像超分损失 5.30改
        num_zeros = 0
        available_points = np.zeros(patch_num)
        # img_show = ipl1[4:-4,4:-4]
        # img_show = img_show/4
        for i in range(patch_num):
            if label[i,0] == 1:
                x_axis = math.floor(index[i]/(ipl1.shape[1]-8))
                y_axis = index[i]%(ipl1.shape[1]-8)
                # img_show[x_axis, y_axis] = 0.5
                if ipl1[x_axis:x_axis + 9, y_axis:y_axis + 9].sum()==0:
                    num_zeros = num_zeros + 1
                    available_points[i] = 1
                    # img_show[x_axis,y_axis] = 1

        if num_zeros != 0:
            batch_t2_up = torch.reshape(batch_t2_up, (patch_num, 9, 9, channels))
            new_t1_mid = torch.zeros(
                (num_zeros, batch_t1_mid.shape[1], batch_t1_mid.shape[2], batch_t1_mid.shape[3])).to(device)
            new_t2_mid = torch.zeros(
                (num_zeros, batch_t1_mid.shape[1], batch_t1_mid.shape[2], batch_t1_mid.shape[3])).to(device)
            if channels == 154:
                new_t1_down = torch.zeros((num_zeros, 27, 27)).to(device)  # 4.13 ???????
                new_pan_t2 = torch.zeros((num_zeros, 27, 27)).to(device)
            else:
                new_t1_down = torch.zeros((num_zeros, 27, 27, 3)).to(device)  # 4.13 ???????
                new_pan_t2 = torch.zeros((num_zeros, 27, 27, 3)).to(device)
            new_hs_t1 = torch.zeros((num_zeros, 9, 9, channels)).to(device)
            new_t2_up = torch.zeros((num_zeros, 9, 9, channels)).to(device)
            num2 = 0
            for num in range(patch_num):
                # if label[num, 1] == 0:
                if available_points[num] == 1:
                    if channels == 224:
                        new_t1_down[num2, :, :, :] = batch_t1_down[num, :, :, :].permute(1, 2, 0)
                        new_pan_t2[num2, :, :, :] = pan_t2[num, :, :, :]
                    else:
                        new_t1_down[num2, :, :] = batch_t1_down[num, 0, :, :]
                        new_pan_t2[num2, :, :] = pan_t2[num, :, :]
                    new_hs_t1[num2, :, :, :] = hs_t1[num, :, :, :]
                    new_t2_up[num2, :, :, :] = batch_t2_up[num, :, :, :]
                    new_t1_mid[num2, :, :, :] = batch_t1_mid[num, :, :, :]
                    new_t2_mid[num2, :, :, :] = batch_t2_mid[num, :, :, :]
                    num2 = num2 + 1

            # 打印超分结果图
            if (batch == 0) & (((e + 1) % 10 == 0) or (e == 0)):
                new_hs_t1_show = new_hs_t1[0, :, :, (10, 20, 35)].detach().cpu().numpy()
                new_t2_up_show = new_t2_up[0, :, :, (10, 20, 35)].detach().cpu().numpy()
                new_t1_down_show = new_t1_down[0, :, :].detach().cpu().numpy()
                new_pan_t2_show = new_pan_t2[0, :, :].detach().cpu().numpy()
                image_t1_sr = new_t1_mid[0, (10, 20, 35), :, :].detach().cpu().numpy()
                image_t1_sr = image_t1_sr.transpose(1, 2, 0)
                image_t2_sr = new_t2_mid[0, (10, 20, 35), :, :].detach().cpu().numpy()
                image_t2_sr = image_t2_sr.transpose(1, 2, 0)

                fig, axs = plt.subplots(2, 3)
                fig.suptitle(name_data + '_epoch:' + str(e + 1) + ',batch:' + str(batch))
                # 显示第1个图像
                axs[0, 0].imshow(new_hs_t1_show)
                axs[0, 0].set_title('t1_input')
                # 显示第2个图像
                axs[1, 0].imshow(new_t2_up_show)
                axs[1, 0].set_title('t2_up')
                # 显示第3个图像
                axs[0, 1].imshow(new_t1_down_show)
                axs[0, 1].set_title('t1_down')
                # 显示第4个图像
                axs[1, 1].imshow(new_pan_t2_show)
                axs[1, 1].set_title('t2_input')
                # 显示第5个图像
                axs[0, 2].imshow(image_t1_sr)
                axs[0, 2].set_title('t1_sr')
                # 显示第6个图像
                axs[1, 2].imshow(image_t2_sr)
                axs[1, 2].set_title('t2_sr')
                plt.show()

            loss_guangpu = loss_L1(new_t2_up,new_hs_t1)
            loss_kongjian = loss_L1(new_pan_t2,new_t1_down)
            loss_double_rs = loss_L1(batch_t1_mid, batch_t2_mid)
            # if e<end_renew_SR:
            #     loss = loss*0.02 + loss_kongjian + loss_guangpu + loss_double_rs*0.2
            # else:
            #     loss = loss + loss_kongjian*0.1 + loss_guangpu*0.1 + loss_double_rs*0.02


            loss = loss*(1-3*alpha) + loss_kongjian*alpha  + loss_guangpu*alpha + loss_double_rs*alpha




        # 损失反向传播与优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss,Sum

    elif grad == 'no_grad':
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


if need_cut_data:
    main_cut()

main()
