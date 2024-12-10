import random

import numpy
import numpy as np
import scipy.io as sio
import data.create_data,data.create_data_USA
import nets
from torch.optim import Adam
import math
import torch
import scipy.sparse as sp
from torch_geometric.data import Data,Batch
import time
import graph_matching
from utils import adjust_learning_rate,build_graph_lower,build_graph_upper,build_graph_middle
import matplotlib.pyplot as plt
from skimage.transform import rescale
from Spatial_SR.common import default_conv
from skfuzzy.cluster import cmeans
from Spatial_SR.SSPSR import SSPSR
from Spectral_SR.model_27 import SpeSRModel
from tqdm import tqdm
from torch.utils.data import DataLoader,Subset
from dataset import AllDataset,Dataset2

pic_title = 'NEW'

name_data = 'China'



# 全局变量
need_val = True                                      # 暂时没设置val
need_train = True                                    # 暂时没写test，只能True
need_cut_data = False                                # 数据预处理

need_Spatial_Super_Resolution = True                 # 空间维超分
need_Spectral_Super_Resolution = True                # 光谱维超分
clustering = True                                   # 聚类
clustering_begin = 10                               # 从第几次迭代开始聚类

model_name = './Spatial_SR/pre_trained_model/pre_USA.pth'  # 预训练空间超分模型
model_name_Spectral = './Spectral_SR/pretrained_model/USA_pretrained.pth' # 预训练光谱超分模型
pseudo_label_dir = './data/pseudo_label_China.mat'          #读入经典算法伪标签

need_save_model = False                              # 是否存模型
lower_pool_or_not = True                             # 下层用27x27还是pool到13x13,默认pool
load_origin_dir = './data/China.mat'
load_patch_dir = 'data/data_change/' + name_data

# 超分里用到的dir在SSR_main里设置吧，简单一点

epochs = 100
batch_size = 32
learning_rate = 1e-4
checkpoint = 300  # 不需要这玩意了现在
channels = 154
gpu = 'cuda:2'
device = torch.device(gpu)
patchsize_HSI = 9
patchsize_PAN = 3 * patchsize_HSI

net_up_size1 = 64  # 上支路第一个MLP的输出尺寸 64 128
net_up_size2 = 128  # 上支路第一个MLP的隐层尺寸 128 256
net_down_size1 = 64 # 下支路第一个CNN的输出尺寸,GCN的输入尺寸 64
mid_hidden_size = 32 # 中路隐层尺寸

# torch.set_printoptions(precision=6)


def main():
    print('======> start loading the image!')

    # 加载T1超分上来的高分高光谱图像（已切patch）和T2的PAN图像（整图）分别构成的patch列表
    if need_cut_data == True:

        if name_data == 'USA':
            print('USA特供版数据预处理')
            list_hs_t1, list_pan_t1, list_mid_t1 = data.create_data_USA.create_HS_list(load_origin_dir, 't1', patchsize_HSI)
            list_hs_t2, list_pan_t2, list_mid_t2 = data.create_data_USA.create_HS_list(load_origin_dir, 't2', patchsize_HSI)
            list_gt, high = data.create_data_USA.create_gt(load_origin_dir)
            list_pseudo_label = data.create_data_USA.create_pseudo_label(pseudo_label_dir)

        else:
            list_hs_t1,list_pan_t1,list_mid_t1 = data.create_data.create_HS_list(load_origin_dir, 't1', patchsize_HSI)
            list_hs_t2,list_pan_t2,list_mid_t2 = data.create_data.create_HS_list(load_origin_dir, 't2', patchsize_HSI)
            list_gt,high = data.create_data.create_gt(load_origin_dir)
            list_pseudo_label =  data.create_data.create_pseudo_label(pseudo_label_dir)

        arr = np.arange(len(list_gt))
        # 打印一下伪标签和真值
        list_pseudo_label2 = np.array(list_pseudo_label)
        equal_count = np.sum(np.equal(list_gt, list_pseudo_label2))
        list_pseudo_label2 = np.reshape(list_pseudo_label2, (high, int(len(list_gt) / high)))
        list_gt2 = np.reshape(list_gt, (high, int(len(list_gt) / high)))
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('pseudo & gt of ' + name_data +',num of same:' + str(equal_count))
        axs[0].imshow(list_gt2, cmap='gray')
        axs[0].set_title('gt')
        axs[1].imshow(list_pseudo_label2, cmap='gray')
        axs[1].set_title('pseudo_label')
        plt.show()

        # shuffle
        state = np.random.get_state()
        np.random.shuffle(list_hs_t1)
        np.random.set_state(state)
        np.random.shuffle(list_hs_t2)
        np.random.set_state(state)
        np.random.shuffle(list_pan_t1)
        np.random.set_state(state)
        np.random.shuffle(list_pan_t2)
        np.random.set_state(state)
        np.random.shuffle(list_mid_t1)
        np.random.set_state(state)
        np.random.shuffle(list_mid_t2)
        np.random.set_state(state)
        np.random.shuffle(list_gt)
        np.random.set_state(state)
        np.random.shuffle(arr)
        np.random.set_state(state)
        np.random.shuffle(list_pseudo_label)

        # 把标签搞成独热码
        list_gt = torch.tensor(list_gt)
        if channels==154:
            list_gt = torch.nn.functional.one_hot(list_gt, 2).type(torch.FloatTensor)
        elif channels == 224:
            list_gt = torch.nn.functional.one_hot(list_gt, 3).type(torch.FloatTensor)
            list_gt[:,[0,1,2]] = list_gt[:,[1,2,0]]
            # 改变后: 0不变 1变 2背景
        else:
            print('unknown number of channels:',channels)
            exit()

        # 造一下超分要用的升采样版list_hs_t1:
        '''# list_upsampling_t1 = [] 旧版本，处理太慢了
        # for img in list_hs_t1:
        #     img_upsampling = rescale(img, (3, 3, 1))
        #     list_upsampling_t1.append(img_upsampling)'''
        time1 = time.time()
        list_upsampling_t1 = []
        for num2 in range(int(len(list_hs_t1) / 6000)):
            if num2 != (int(len(list_hs_t1) / 6000) - 1):
                list_hs_t1_small = list_hs_t1[(num2 * 6000):(num2 * 6000 + 6000)]
            else:
                list_hs_t1_small = list_hs_t1[(num2 * 6000):]
            numpy_hs_t1 = np.array(list_hs_t1_small)
            numpy_hs_t1 = numpy_hs_t1.transpose((0, 3, 1, 2))
            numpy_hs_t1 = torch.Tensor(numpy_hs_t1).to(gpu)
            numpy_hs_t1 = torch.nn.functional.interpolate(numpy_hs_t1, scale_factor=(3, 3), mode='bilinear')
            numpy_hs_t1 = numpy_hs_t1.detach().cpu().numpy()
            numpy_hs_t1 = numpy_hs_t1.transpose((0, 2, 3, 1))
            for num in range(numpy_hs_t1.shape[0]):
                list_upsampling_t1.append(numpy_hs_t1[num, :, :, :])
        print('对list_hs_t1进行升采样,耗时', time.time() - time1, '，总升采样patch数', len(list_upsampling_t1))

        print('开始切patch')
        print('patch save_dit:./data/data_change/'+name_data+'/')
        time1 = time.time()
        time_all = time.time()
        for i in range(len(list_gt)):
            mdict = {'hs_t1':list_hs_t1[i],'hs_t2':list_hs_t2[i],'pan_t1':list_pan_t1[i],'pan_t2':list_pan_t2[i],
                     'mid_t1':list_mid_t1[i],'mid_t2':list_mid_t2[i],'gt':list_gt.numpy()[i,:],'arr':arr[i],'pseudo_label':list_pseudo_label[i],
                     'upsample_t1':list_upsampling_t1[i]}
            sio.savemat('./data/data_change/'+name_data+'/'+str(i+1)+'.mat',mdict=mdict)
            if (i+1)%1000==0:
                print(i+1,'patch已切,1000个patch用时',time.time()-time1)
                time1 = time.time()
        print('已切完,总用时',time.time()-time_all)

    # 将patch读入dataloader
    all_dataset = AllDataset(load_patch_dir)
    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('loading dataset,dir = .', load_patch_dir)
    gt, height = data.create_data.create_gt(load_origin_dir)
    width = int(len(gt) / height)
    print('数据点总数:{},图片尺寸:{}x{}'.format(len(gt),height,width))

    # 图匹配网
    nets_lower = [nets.CNN(net_size1=net_up_size1),nets.GCN(net_size1=net_up_size1),nets.message_net(),nets.MLP(),nets.MLP2(),nets.MLP3()]
    nets_upper = [nets.MLP0(channels=channels, net_size1=net_up_size1,net_size2=net_up_size2),
                  nets.GCNup(patchsize=patchsize_HSI), nets.message_netup(), nets.MLPup(), nets.MLP2up(), nets.MLP3()]
    nets_middle = [
        nets.CNNmid(channel=channels,patchsize=patchsize_HSI,down_size=net_down_size1, out_size = net_up_size1,hidden_size=mid_hidden_size),
        nets.GCNmid(),nets.InnerpAffinity(d=64),nets.SpectralMatching(),nets.Sinkhorn(),nets.MLPmid(),nets.LINSHI()
    ]  # 阿巴阿巴阿巴巴



    # 空间超分网
    nets_SSPSR = SSPSR(n_subs=8, n_ovls=1, n_colors=channels, n_blocks=3, n_feats=256,
            n_scale=3, res_scale=0.1, use_share=True, conv=default_conv)
    state_dict = torch.load(model_name)
    nets_SSPSR.load_state_dict(state_dict)
    if need_Spatial_Super_Resolution == True:
        print('use pre_trained SSPSR model name:', model_name)
    nets_SSPSR.to(device)
    """
    "n_subs, default set to 8")  # 每组光谱带数
    "n_ovls, default set to 1")  # 光谱带间的重叠
    "n_scale, default set to 3")  # 超分比例
    "n_blocks, default set to 6")  # 空间频谱快数量
    """

    # 光谱超分网
    nets_SpeSR = SpeSRModel(input_features=1, output_features=channels)
    state_dict2 = torch.load(model_name_Spectral)
    nets_SpeSR.load_state_dict(state_dict2)
    if need_Spectral_Super_Resolution == True:
        print('use pre_trained SpeSR model name:', model_name_Spectral)
    nets_SpeSR.to(device)

    # 参数更新
    parameters = []
    # 三个图匹配网
    for net in nets_upper:
        net.to(device)
        parameters = parameters + list(net.parameters())
    for net in nets_lower:
        net.to(device)
        parameters = parameters + list(net.parameters())
    for net in nets_middle:
        net.to(device)
        parameters = parameters + list(net.parameters())
    # 两个超分网
    parameters = parameters + list(nets_SSPSR.parameters())
    parameters = parameters + list(nets_SpeSR.parameters())

    # 损失与优化器
    loss_classification = torch.nn.CrossEntropyLoss().to(device)
    loss_MSE = torch.nn.MSELoss().to(device)
    optimizer = Adam(parameters , lr=learning_rate, weight_decay=0)

    # 对于每次迭代
    if(need_train == True):
        print('============> start training!')
        for e in range(epochs):
            # 对于每个batch
            train_or_var('train', nets_upper, nets_lower, nets_middle,nets_SSPSR,nets_SpeSR, all_dataset,all_dataloader,
                         e,loss_classification, loss_MSE, optimizer, height,width)


    # 保存模型
    """
    if (need_save_model == True):
        k = 0
        for net in nets_upper:
            torch.save(net.state_dict(), './save_model/nets' + str(k) + '.pth')
            k = k + 1
        k = 0
        for net in nets_lower:
            torch.save(net.state_dict(), './save_model/nets_lower' + str(k) + '.pth')
            k = k + 1
        k = 0
        for net in nets_lower:
            torch.save(net.state_dict(), './save_model/nets_middle' + str(k) + '.pth')
            k = k + 1"""


def train_or_var(type, nets_upper, nets_lower, nets_middle, nets_SSR, nets_SpeSR, all_dataset, all_dataloader, e,
                 loss_classification, loss_MSE, optimizer, height,width):
    time_begin = time.time()
    print('============> <{}/{}>epoch is beginning'.format(e + 1, epochs))
    adjust_learning_rate(learning_rate, optimizer, e + 1)
    print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
    for net in nets_upper:
        net.train()
    for net in nets_lower:
        net.train()
    for net in nets_middle:
        net.train()
    nets_SSR.train()
    nets_SpeSR.train()

    # 本次迭代内变量
    loss_sum = 0                                  # 本次迭代总loss
    correct_sum = 0                               # 本次迭代总正确分类数
    dif = torch.zeros((height*width, 256))        # 上下路差异向量，用于聚类
    distances = torch.zeros(height*width)         # 中路batch间欧氏距离
    result = np.zeros(height*width)               # 分类结果
    gt_for_show = np.zeros(height*width)          # 用于展示的标签
    gt = np.zeros(height*width)                   # 真实标签，用于计算正确率
    weibiaoqian = np.zeros(height*width)          # 传统算法或聚类法所得伪标签集
    pseudo_labels = np.zeros(height*width)        # 传统算法全图伪标签
    arrs = np.zeros(height*width).astype(int)

    # 更新聚类和伪标签
    with torch.no_grad():
        time_no_loss_begin = time.time()
        for batch,(hs_t1,hs_t2,pan_t1,pan_t2,mid_t1,mid_t2,list_gt,arr,pseudo_label,upsample_t1) in enumerate(all_dataloader):
            # if (batch % 50 == 0):
            #     time_batch_begin = time.time()
            dif,correct,result,gt_for_show,distances = batch_train_or_var(
                type,'no_grad',batch,hs_t1,upsample_t1,hs_t2,pan_t1,pan_t2,mid_t1,mid_t2,
                nets_upper,nets_lower,nets_middle,nets_SSR,nets_SpeSR,
                dif=dif,list_gt=list_gt,arr=arr,result=result,gt=gt_for_show,distances=distances)
            pseudo_labels[batch*batch_size:batch*batch_size+len(hs_t1)] = pseudo_label
            gt[batch*batch_size:batch*batch_size+len(hs_t1)] = list_gt[:,1]
            arrs[batch*batch_size:batch*batch_size+len(hs_t1)] = arr
            correct_sum = correct_sum + correct
            # if ((batch+1) % 50 == 0):
            #     print('<',batch+1,'/',len(all_dataloader),'>batch,time for last 50 batch:',time.time()-time_batch_begin)
        print('batch num for no_loss iteration:',batch+1,'.times:',time.time()-time_no_loss_begin)
    # 将本次迭代最终版的dif和distances转为numpy
    dif = dif.cpu().numpy()
    distances = distances.cpu().numpy()


    # 聚类算法求伪标签
    if (clustering == True)&(e>=clustering_begin):
        cntr,u,u0,d,jm,p,fpc = cmeans(dif.T,c=2,m=2,error=0.005,maxiter=1000)
        del cntr,u0,d,jm,p,fpc
        """
                    cntr：一个形状为 (c, m) 的二维数组，表示聚类的簇中心。
                    u：一个形状为 (c, n) 的二维数组，表示每个数据点与各个簇的隶属度。
                    u0：一个形状为 (c, n) 的二维数组，表示迭代过程中每个数据点与各个簇的隶属度。
                    d：一个形状为 (c, n) 的二维数组，表示每个数据点到各个簇中心的距离。
                    jm：一个实数，表示目标函数值（objective function value）。
                    p：一个整数，表示迭代次数。
                    fpc：一个实数，表示模糊分区系数（fuzzy partition coefficient）。
                    参数c为簇数，m为模糊度，越小越严格，error为容错率，控制收敛，maxiter为最大迭代次数
                    """
        # 取置信度最高的20%作为伪标签
        n_clusters,n_samples = u.shape
        labels_pred = np.argmax(u, axis=0)
        confidence = np.max(u, axis=0)

        num_zeros = np.sum(labels_pred==0)
        num_ones = np.sum(labels_pred==1)
        if num_ones>num_zeros:
            labels_pred = 1 - labels_pred

        if e<5 :
            Proportion =0.1
        else:
            Proportion =0.2

        top_indice = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels_pred == i)[0]
            cluster_confidences = confidence[cluster_indices]
            # top_indice.append(cluster_indices[np.argsort(cluster_confidences)[::-1][:int(len(cluster_indices) * Proportion)]])
            top_indice.append(cluster_indices[np.argsort(cluster_confidences)[::-1][:int(n_samples * Proportion * 0.5)]])
        top_indices =  np.concatenate((top_indice[0], top_indice[1]), axis=0)
        pseudo_labels = labels_pred[top_indices]
        gt_labels = gt[top_indices]

    # 传统算法求伪标签
    if (clustering == False) or (e<clustering_begin):
        # 欧氏距离构造伪标签
        """
        # n = int(distances.shape[0] * 0.2)
        # max_indices = np.argsort(distances)[-n:]
        # max_indices = np.random.choice(max_indices, size=int(len(max_indices) * 0.5), replace=False)
        # min_indices = np.argsort(distances)[:n]
        # min_indices = np.random.choice(min_indices, size=int(len(min_indices) * 0.5), replace=False)
        # top_indices = np.concatenate((max_indices, min_indices),axis=0)
        # threshold = np.quantile(distances, 0.5)
        # labels_pred = np.zeros_like(distances).astype(int)
        # labels_pred[distances >= threshold] = 1
        # pseudo_labels = labels_pred[top_indices]
        # gt_labels = (list_gt[:, 1].numpy())[top_indices]
        """
        # 经典方法构造伪标签
        labels_pred = pseudo_labels
        zero_indices = [i for i, x in enumerate(labels_pred) if x == 0]
        one_indices = [i for i, x in enumerate(labels_pred) if x == 1]
        zero_sample = random.sample(zero_indices, k=int(len(labels_pred) * 0.1))
        one_sample = random.sample(one_indices, k=int(len(labels_pred) * 0.1))
        top_indices = zero_sample + one_sample
        pseudo_labels = [labels_pred[i] for i in top_indices]
        labels_pred = np.array(labels_pred)
        pseudo_labels = np.array(pseudo_labels)
        top_indices = np.array(top_indices)
        gt_labels = gt[top_indices]

    # 看一看伪标签和真实标签的差异
    gt_count = np.sum(gt)
    print('真实标签中为1的标签数量{},为0的标签数量{}'.format(gt_count,len(gt)-gt_count))
    pseudo_count = np.sum(labels_pred)
    print('伪标签中为1的标签数量{},为0的标签数量{}'.format(pseudo_count,len(labels_pred)-pseudo_count))

    oo,ol,lo,ll = 0,0,0,0
    # print(pseudo_labels,'\n',gt_labels)
    for i in range(len(gt_labels)):
        if (gt_labels[i]==0) & (pseudo_labels[i]==0):
            oo = oo+1
            weibiaoqian[arrs[top_indices[i]]] = 1
        elif (gt_labels[i]==0) & (pseudo_labels[i]==1):
            ol = ol+1
            weibiaoqian[arrs[top_indices[i]]] = 0.5
        elif (gt_labels[i]==1) & (pseudo_labels[i]==0):
            lo = lo+1
            weibiaoqian[arrs[top_indices[i]]] = 0.5
        elif (gt_labels[i]==1) & (pseudo_labels[i]==1):
            ll = ll+1
            weibiaoqian[arrs[top_indices[i]]] = 1
    print('total:{},<真0伪0>:{},<真0伪1>:{},<真1伪0>:{},<真1伪1>:{}'.format(len(gt_labels),oo,ol,lo,ll))

    perm = np.random.permutation(len(top_indices))
    nums,labels =  top_indices[perm], pseudo_labels[perm]
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(np.eye(2)[labels.astype(int)])

    # 用伪标签更新模型
    subset = Subset(all_dataset,nums)
    sub_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    time_with_loss_begin = time.time()
    for batch2,(hs_t1,hs_t2,pan_t1,pan_t2,mid_t1,mid_t2,list_gt,arr,pseudo_label,upsample_t1) in enumerate(sub_dataloader):
        # if ((batch) % 50 == 0):
        #     time_batch_begin = time.time()
        loss = batch_train_or_var(
            type, 'with_grad', batch2,hs_t1,upsample_t1,hs_t2,pan_t1,pan_t2,mid_t1,mid_t2,
            nets_upper, nets_lower, nets_middle, nets_SSR, nets_SpeSR,
            labels=labels, loss_classification=loss_classification, loss_MSE=loss_MSE, optimizer=optimizer)
        loss_sum = loss_sum + loss
        # if ((batch + 1) % 50 == 0):
        #     print('<', batch + 1, '/', len(sub_dataloader), '>batch,time for last 50 batch:',
        #           time.time() - time_batch_begin)
    print('batch num for with_loss iteration:', batch2 + 1, '.times:', time.time() - time_with_loss_begin)

    # 打印结果图
    result = np.reshape(result, (height, width))
    gt_for_show = np.reshape(gt_for_show, (height,width))
    weibiaoqian = np.reshape(weibiaoqian, (height,width))
    labels_pred = np.reshape(labels_pred[np.argsort(arrs)], (height,width))
    distances = distances[np.argsort(arrs)]
    distances = np.reshape(distances, (height,width))
    fig, axs = plt.subplots(1, 5)
    fig.suptitle(str(e+1)+' '+pic_title+'\n'+'accuracy:'+str(correct_sum / (width*height)))
    # 显示第1个图像
    axs[0].imshow(gt_for_show, cmap='gray')
    axs[0].set_title(str(e + 1) + ' gt_for_show')
    # 显示第2个图像
    axs[1].imshow(weibiaoqian, cmap='gray')
    axs[1].set_title(str(e + 1) + ' weibiaoqian')
    # 显示第3个图像
    axs[2].imshow(labels_pred, cmap='gray')
    axs[2].set_title(str(e + 1) + ' cluster result')
    # 显示第4个图像
    axs[3].imshow(distances, cmap='gray')
    axs[3].set_title(str(e + 1) + ' distance')
    # 显示第5个图像
    axs[4].imshow(result, cmap='gray')
    axs[4].set_title(str(e + 1) + ' result')
    plt.show()

    print(
        '<{}/{}>epoch has done,Time for this epoch: {:.2f} seconds,avg loss in this epoch：{:.6f},avg correct in this epoch：{:.6f}'
        .format(e + 1, epochs, time.time() - time_begin, loss_sum / len(sub_dataloader), correct_sum / (width*height)))




def batch_train_or_var(
        type,grad,batch,hs_t1,upsample_t1,hs_t2,pan_t1,pan_t2,mid_t1,mid_t2,nets_upper,nets_lower,nets_middle,nets_SSR,nets_SpeSR,
        dif=None, labels=None,loss_classification=None,loss_MSE=None,optimizer=None,list_gt=None,arr=None,result=None,gt=None,distances=None):

    # 最后一个batch的batch_num可能会小于batch_size，其余时刻相等
    patch_num = hs_t1.shape[0]

    # 将每个patch加载进batch 现在直接就是batch了所以没必要咯 之前的语句删了都
    # 上路
    batch_t1_up_no_reshape = hs_t1.numpy()
    batch_t1_up = torch.reshape(hs_t1,(-1,channels)).to(device)
    batch_t2_up = torch.reshape(hs_t2,(-1,channels)).to(device)
    batch_up_sampling = upsample_t1.numpy()
    batch_up_sampling = numpy.transpose(batch_up_sampling, (0, 3, 1, 2))
    batch_up_sampling = torch.from_numpy(batch_up_sampling).to(device)
    # 中路
    batch_t1_mid = mid_t1.numpy()
    batch_t1_mid = numpy.transpose(batch_t1_mid,(0,3,1,2))
    batch_t1_mid = torch.from_numpy(batch_t1_mid).to(device)
    batch_t2_mid = mid_t2.numpy()
    batch_t2_mid = numpy.transpose(batch_t2_mid,(0,3,1,2))
    batch_t2_mid = torch.from_numpy(batch_t2_mid).to(device)
    # 下路
    batch_t1_down = pan_t1.unsqueeze(1).to(device)
    batch_t2_down = pan_t2.unsqueeze(1).to(device)
    del hs_t1,hs_t2,mid_t1,mid_t2,pan_t1,pan_t2,upsample_t1



    #对T1时刻图像进行空间维超分！！！！！！！！超超超
    if need_Spatial_Super_Resolution == True:
        batch_t1_up_no_reshape = batch_t1_up_no_reshape.transpose(0,3,1,2)
        batch_t1_up_no_reshape = torch.from_numpy(batch_t1_up_no_reshape).to(device)
        batch_t1_mid = nets_SSR(batch_t1_up_no_reshape,batch_up_sampling)
        batch_t1_down = torch.mean(batch_t1_mid, dim=1).unsqueeze(1)


    #对T2时刻图像进行光谱维超分！！！！！！！！超超超
    if need_Spectral_Super_Resolution == True:
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

    batch_t1_up_afterCNN = batch_t1_up_afterCNN.reshape((patch_num, patchsize_HSI * patchsize_HSI, net_up_size1))
    batch_t2_up_afterCNN = batch_t2_up_afterCNN.reshape((patch_num, patchsize_HSI * patchsize_HSI, net_up_size1))

    # 每个patch构一张图，每个batch的图构成一个列表
    list_graph_up = []
    list_graph_down = []
    tensor_up_t1 = torch.zeros((patch_num, net_up_size1, patchsize_HSI * patchsize_HSI)).to(device)
    tensor_up_t2 = torch.zeros((patch_num, net_up_size1, patchsize_HSI * patchsize_HSI)).to(device)
    if lower_pool_or_not == False:
        tensor_down_t1 = torch.zeros((patch_num, 729, net_down_size1)).to(device)
        tensor_down_t2 = torch.zeros((patch_num, 729, net_down_size1)).to(device)
    else:
        tensor_down_t1 = torch.zeros((patch_num, 169, net_down_size1)).to(
            device)  # 169我懒得写成变量形式了，总之是向下取整(patchsize_PAN/2)的平方
        tensor_down_t2 = torch.zeros((patch_num, 169, net_down_size1)).to(device)
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
    batch_t1_mid_afterCNN = nets_middle[0](batch_t1_mid, tensor_up_t1, tensor_down_t1, lower_pool_or_not)
    batch_t2_mid_afterCNN = nets_middle[0](batch_t2_mid, tensor_up_t1, tensor_down_t2, lower_pool_or_not)
    graph_t1_mid, A_t1, top_k = build_graph_middle(batch_t1_mid_afterCNN, device)  # 邻接矩阵在图匹配中还要用
    graph_t2_mid, A_t2, _ = build_graph_middle(batch_t2_mid_afterCNN, device)
    graph_t1_mid = graph_t1_mid.to(device)
    graph_t2_mid = graph_t2_mid.to(device)

    # 三路图匹配
    preds_up, dif_up = graph_matching.graph_matching_UPorDOWN(batch_graph_up, nets_upper, patch_num)
    preds_down, dif_down = graph_matching.graph_matching_UPorDOWN(batch_graph_down, nets_lower, patch_num)
    # feature augmentation vector 中路图匹配得特征增强向量
    aug_vec = graph_matching.graph_matching_middle(graph_t1_mid, graph_t2_mid, A_t1, A_t2, nets_middle, top_k, device)
    dif_mid = torch.cat((dif_up, dif_down), dim=1)
    dif_mid = dif_mid.transpose(1, 0)
    dif_mid = torch.mul(dif_mid, aug_vec).transpose(1, 0)
    preds_mid = nets_middle[5](dif_mid)

    """一些看中间结果的函数总之暂时不用封存起来
    # if type == 'train':
    #     if (batch == 6) & ((e + 1) % 10 == 0):
    #         # print('graph1.x',graph1.x,'\ngraph2.x',graph2.x,'\nMp:',Mp,'\ns:',s)
    #         labels_re = labels[:, 0].cpu().detach().numpy()
    #         sio.savemat('./report/batch' + str(batch) + 'epoch' + str(e) + '_gt.mat', mdict={'labels': labels_re})

    # if type == 'train':
    #     if batch == 6:
    #         for iii in range(3):
    #             abcdefg = []
    #             abcdefg.append(int(labels[iii,0].item()))
    #             if loss_type2 == 'MSE':
    #                 abcdefg.append(round((preds_mid[iii]).item(), 4))
    #             else:
    #                 abcdefg.append(round((preds_mid[iii,0]).item(),4))
    #                 abcdefg.append(round((preds_mid[iii,1]).item(),4))
    #             print(abcdefg)
    #     # if batch == 6:
    #     #     for iii in range(aug_vec.shape[0]):
    #     #         abcdefg = []
    #     #         abcdefg.append(int(labels[iii,0].item()))
    #     #         abcdefg.append(round((aug_vec[iii]).item(),4))
    #     #         abcdefg.append(round((preds_mid[iii,0]).item(),4))
    #     #         print(abcdefg)
    """

    if (type=='train') & (grad == 'with_grad'):
        # 分类损失
        preds_complex = torch.cat((preds_up, preds_down,preds_mid), dim=0)
        label = labels[(batch * batch_size):(batch * batch_size + patch_num), :].to(device)
        label_complex = torch.cat((label,label,label), dim=0)
        loss1 = loss_classification(preds_complex, label_complex)
        loss = loss1

        # 中路不变图差异损失
        num_zeros = int((label[:, 1].numel() - label[:, 1].sum()).item())
        if num_zeros != 0:
            new_t1_mid = torch.zeros((num_zeros,batch_t1_mid.shape[1],batch_t1_mid.shape[2],batch_t1_mid.shape[3])).to(device)
            new_t2_mid = torch.zeros((num_zeros,batch_t1_mid.shape[1],batch_t1_mid.shape[2],batch_t1_mid.shape[3])).to(device)
            num2 = 0
            for num in range(patch_num):
                if label[num,1] == 0:
                    new_t1_mid[num2,:,:,:] = batch_t1_mid[num,:,:,:]
                    new_t2_mid[num2,:,:,:] = batch_t2_mid[num,:,:,:]
                    num2 = num2 + 1

            loss2 = loss_MSE(batch_t1_mid, batch_t2_mid)*10
            loss = loss + loss2

            if(batch%30)==0:
                print('batch',batch+1,',loss_分类器:',loss1.item(),',loss_中路图片*10:',loss2.item())

        # 损失反向传播与优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    if grad == 'no_grad':
        # 上下路差异向量
        dif_complex = torch.cat((dif_up, dif_down), dim=1)
        dif[(batch * batch_size):(batch * batch_size + patch_num),:] = dif_complex.detach()

        # 三路预测向量
        _, predicted_complex = torch.max((preds_down + preds_up + preds_mid) / 3, 1)
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
        batch_t1_mid = batch_t1_mid.view(patch_num,-1)
        batch_t1_mid = batch_t1_mid / torch.norm(batch_t1_mid,dim=0,keepdim=True)
        batch_t2_mid = batch_t2_mid.view(patch_num,-1)
        batch_t2_mid = batch_t2_mid / torch.norm(batch_t2_mid,dim=0,keepdim=True)
        distance = torch.sqrt(torch.sum((batch_t1_mid - batch_t2_mid) ** 2, dim=1))

        # 中路欧氏距离向量
        distances[(batch * batch_size):(batch * batch_size + patch_num)] = distance
        return dif,correct,result,gt,distances



main()
