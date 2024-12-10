from __future__ import print_function
from __future__ import division
import warnings
warnings.filterwarnings("ignore")

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np
import math
from scipy.io import savemat
# import 自定义模块
import random
from my_dataset import traditional_method_results
import slic_simple
# 设定随机数种子



def divide(gt):
    change = []
    unchange = []
    change_xy = np.where(gt == 1)  ##返回满足像素值等于1的行列号，change_xy[0]为行，change_xy[1]为列
    unchange_xy = np.where(gt == 0)

    change = list(change_xy)
    unchange = list(unchange_xy)

    return change,unchange

def rand_select_indices(gt,change_gt, unchange_gt):
    change_gt = [list(t) for t in zip(*change_gt)]
    unchange_gt = [list(t) for t in zip(*unchange_gt)]

    change_train= random.sample(change_gt, int(0.2* len(change_gt)))
    # change_test=[(x,y) for (x,y) in change_gt if (x,y) not in change_train]

    unchange_train = random.sample(unchange_gt, int(0.2 * len(unchange_gt)))
    # unchange_test=[(x,y) for (x,y) in unchange_gt if (x,y) not in unchange_train]

    train_indices=change_train+unchange_train
    h,w=gt.shape
    test_indices=[(x,y) for x in range(h) for y in range(w)]

    return train_indices, test_indices



def image_into_patch(image_T1, image_T2,class_count, superpixel_scale):
    """
    transform T1 and T2 into patch
    """
    height, width, band = image_T1.shape
    img = np.concatenate((image_T1, image_T2), axis=-1)  ##T1和T2级联

    ls = slic_simple.LDA_SLIC(img, class_count)
    Q1, S1, A1, superpixel_scount = ls.simple_superpixel(scale=superpixel_scale)  ### 超像素分割
    # Q 输入图和超像素之间的关系
    # S 初始化A的时候使用
    # A 邻接矩阵

    image_T1 = image_T1.reshape(-1, band)
    image_T2 = image_T2.reshape(-1, band)
    patch_T1 = np.dot(Q1.T, image_T1) / np.sum(Q1, axis=0).reshape(Q1.shape[1], 1).repeat(band, axis=1)
    patch_T2 = np.dot(Q1.T, image_T2) / np.sum(Q1, axis=0).reshape(Q1.shape[1], 1).repeat(band, axis=1)

    return Q1, S1, A1, superpixel_scount, patch_T1, patch_T2

def clustering(input, components):
    # height, width = input.shape
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(input)
    output = kmeans.predict(input)
    count = Counter(output)

    least_index = min(count, key=count.get)
    # change_map = np.reshape(output, (height, width))
    output[output == least_index] = 255
    output[output != 255] = 0
    output=output/255
    return output

def PCA_Kmeans_detection(T1,T2):
    height, width, bands = T1.shape  # 原始高光谱数据的三个维度
    data1 = np.reshape(T1, [height * width, bands])  # 将数据转为HW * B
    data2 = np.reshape(T2, [height * width, bands])  # 将数据转为HW * B
    diff_image = abs(data1 - data2)  # (58800, 154)
    pca = PCA(n_components=0.99)
    PCA_out = pca.fit_transform(diff_image)
    change_map = clustering(PCA_out, 2)
    output = np.reshape(change_map, [height, width])  # [H W]
    return output



def SLIC_CVA_detection(T1,T2):

    diff_image = T2 - T1  #[ N C]
    diff_image = np.sqrt(np.sum(np.power(diff_image,2),1))  # [ N C]
    N= len(diff_image)
    diff_image = np.reshape(diff_image, [N, 1])  # 将数据转为HW * 1
    kmeans = KMeans(2, verbose=0)       #创建分类器对象
    kmeans.fit(diff_image)             ##用训练数据拟合分类器模型
    output = kmeans.predict(diff_image)   ##用训好的分类器去预测CVA_reshape的标签   [N 1]
    return output

def CVA_detection(T1,T2):
    diff_image = T2 - T1  #[ H W C]
    CVA_map=np.sqrt(np.sum(np.power(diff_image,2),2))   # [H W]
    height, width= CVA_map.shape
    CVA_reshape = np.reshape(CVA_map, [height * width, 1])  # 将数据转为HW * 1
    kmeans = KMeans(2, verbose=0)       #创建分类器对象
    kmeans.fit(CVA_reshape)             ##用训练数据拟合分类器模型
    output = kmeans.predict(CVA_reshape)   ##用训好的分类器去预测CVA_reshape的标签   H*W
    count = Counter(output)
    least_index = min(count, key=count.get)

    # output_reshape = np.reshape(output, [height,width])   #[H W]
    output[output == least_index] = 1
    output[output != 1] = 0
    return output

def get_train_index(Q,CVA_gt,PCA_gt,IRMAD_gt,width):
    one_n=zero_n=0
    Q_T = Q.T
    _, N = Q.shape
    cva_gt_index = []
    one=[]
    zero=[]
    for i in range(N):
        pixels = np.where(Q_T[i, :] == 1)
        CVA_count = np.sum(CVA_gt[pixels[0]])
        PCA_count = np.sum(PCA_gt[pixels[0]])
        IRMAD_count = np.sum(IRMAD_gt[pixels[0]])
        length=len(pixels[0])
        # if (CVA_count >= 0.95*len(pixels[0])) and (PCA_count >= 0.95*len(pixels[0])) and (IRMAD_count >= 0.95*len(pixels[0])):
        #     one_n = one_n + 1
        if (CVA_count == len(pixels[0])) and (PCA_count ==len(pixels[0])) and (IRMAD_count == len(pixels[0])):
            for j in pixels[0]:
                if CVA_gt[j] == 1 or PCA_gt[j] == 1 or IRMAD_gt[j] == 1:
                    one.extend([j])

            # one.extend(list(pixels[0]))
            # one_n = one_n + 1
        elif CVA_count == 0 and PCA_count == 0 and IRMAD_count == 0:
             zero.extend(list(pixels[0]))
             zero_n = zero_n + 1
        # elif (CVA_count <= 0.2*len(pixels[0])) and (PCA_count <= 0.2*len(pixels[0])) and (IRMAD_count <= 0.2*len(pixels[0])):
        #     zero_n = zero_n + 1
        #     for j in pixels[0]:
        #         if CVA_gt[j] == 0 or PCA_gt[j] == 0 or IRMAD_gt[j] == 0:
        #             zero.extend([j])


        # elif count >= 0.95 * len(pixels[0]):
        #     for j in pixels[0]:
        #         if cva_out[j] == 1:
        #             one.append(j)
        #     cva_gt_index.extend(list(one))
        # elif count <= 0.05 * len(pixels[0]):
        #     for j in pixels[0]:
        #         if cva_out[j] == 0:
        #             zero.append(j)
        #     cva_gt_index.extend(list(zero))

    zero = random.sample(zero,int(len(one)))
    cva_gt_index=zero+one
    gt_index = []
    for data in cva_gt_index:
        if data < width:
            gt_index_x = 0
            gt_index_y = data
            gt_index.append([gt_index_x, gt_index_y])
        else:
            gt_index_x = math.floor(data / width)
            gt_index_y = data - gt_index_x * width
            gt_index.append([gt_index_x, gt_index_y])
    return gt_index


def metrics(prediction, target):  # 输出指标
    prediction = prediction.flatten()
    target = target.flatten()

    results = {} #将输出结果定义为字典dictionary类型，作为函数的返回值。
                 # 一开始的时候只是将results作为空字典，然后逐步增加键值对


    ##计算混淆矩阵  输入：Ground truth   分类器返回的估计目标
    cm = confusion_matrix(target,prediction)

    results["Confusion matrix"] = cm

    # 计算分类准确率
    total = np.sum(cm)  ###通过对混淆矩阵每个元素求和，得到总样本数
    accuracy = sum([cm[x][x] for x in range(len(cm))])  #分类准确的样本数
                                                       # 也即是混淆矩阵的对角线元素的总和
    accuracy *= 100 / float(total)
    results["Accuracy"] = accuracy


    # 计算 kappa coefficient

    n = np.sum(cm)
    sum_po = 0
    sum_pe = 0
    for i in range(len(cm[0])):
        sum_po += cm[i][i]
        row = np.sum(cm[i, :])
        col = np.sum(cm[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    kappa=(po - pe) / (1 - pe)
    results["Kappa"]=kappa


    return results

def show_results(results):
    text = ""

    cm = results["Confusion matrix"]
    accuracy = results["Accuracy"]
    kappa = results["Kappa"]


    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"


    text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"


    text += "Kappa: {:.04f}\n".format(kappa)

    # vis.text(text.replace('\n', '<br/>'))
    print(text)
