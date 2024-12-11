import numpy
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from skimage.transform import rescale
import matplotlib.pyplot as plt

def create_HS_list(load_pan_dir, t, patchsize_HSI):
    print('===>>>处理{}时刻图片'.format(t))
    # 读入图片
    datas = sio.loadmat(load_pan_dir)
    if(t=='t2'):
        img = np.array(datas['T2'][...], dtype=np.float32)
    elif(t=='t1'):
        img = np.array(datas['T1'][...], dtype=np.float32)
    else:
        exit()
    img = normalize_data(img)
    bands = img.shape[2]

    # 填充到423x144
    temp = img[:, :, 0]  # 420x140
    temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
    [h_pad, w_pad] = temp2.shape
    img_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = img[:, :, i]
        temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
        img_pad[:, :, i] = temp2
    print('填充后原图片尺寸:',img_pad.shape)

    # img_pad = img
    # print('不填充图片')

    height, width = img_pad.shape[0], img_pad.shape[1]

    # 原图的缩放
    img_lr = rescale(img_pad, (1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
    # 切patch（多光谱）
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = img_lr[:, :, 0]
    temp_mid = img_pad[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = np.int32(pad_width)
    pad_width_mid = np.floor(patchsize_HSI * 3 / 2)
    pad_width_mid = np.int32(pad_width_mid)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    temp2_mid = np.pad(temp_mid, pad_width_mid-1, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    [h_pad_mid, w_pad_mid] = temp2_mid.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    data_mid_pad = np.empty((h_pad_mid, w_pad_mid, bands), dtype='float32')
    for i in range(bands):
        temp = img_lr[:, :, i]
        temp_mid = img_pad[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        temp2_mid = np.pad(temp_mid, pad_width_mid-1, 'symmetric')
        data_HSI_pad[:, :, i] = temp2
        data_mid_pad[:, :, i] = temp2_mid
    print('二次填充后图片尺寸:',data_mid_pad.shape)
    # 构建高光谱patch序列
    img_list_LRHS=[]
    img_list_mid =[]
    k = 0
    for i in range(int(height/3)):
        for j in range(int(width/3)):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = data_HSI_pad[(i):(i + 2 * pad_width + 1),
                    (j):(j + 2 * pad_width + 1),:]
            patch_mid = data_mid_pad[(i*3):(i*3 + 2 * pad_width_mid + 1),
                    (j*3):(j*3 + 2 * pad_width_mid + 1),:]
            # print(patch.shape,patch_mid.shape)
            img_list_LRHS.append(patch)
            img_list_mid.append(patch_mid)
            k = k + 1
    print('每个高光谱低分辨率图的patch的尺寸：', img_list_LRHS[0].shape)
    print('每个高光谱高分辨率图的patch的尺寸：', img_list_mid[0].shape)

    # 将pad后的原图构建为PAN
    img_pan = np.mean(img_pad[:,:,:27], axis=2)
    # img_pan = mat2gray(img_pan)

    # 切patch（PAN）
    pad_width = np.floor(patchsize_HSI * 3 / 2)
    pad_width = np.int32(pad_width)
    img_pan = np.pad(img_pan, pad_width-1, 'symmetric')
    # 构建PAN patch序列
    img_list_PAN=[]
    k = 0
    for i in range(int(height/3)):
        for j in range(int(width/3)):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = img_pan[(3*i):(3*i + 2 * pad_width + 1),
                    (3*j):(3*j + 2 * pad_width + 1)]
            img_list_PAN.append(patch)
            k = k + 1
    print('每个PAN图的patch的尺寸：', img_list_PAN[0].shape)
    # print('每类各', k, '个patch已切完')
    return img_list_LRHS, img_list_PAN, img_list_mid

def create_HS_list_224(load_pan_dir, t, patchsize_HSI):
    print('===>>>处理{}时刻图片'.format(t))
    # 读入图片
    datas = sio.loadmat(load_pan_dir)
    if(t=='t2'):
        img = np.array(datas['T2'][...], dtype=np.float32)
    elif(t=='t1'):
        img = np.array(datas['T1'][...], dtype=np.float32)
    else:
        exit()

    if load_pan_dir == './data/Bay.mat':
        img = normalize_data_2(img)
    else:
        img = normalize_data(img)
    bands = img.shape[2]

    # 填充到423x144
    temp = img[:, :, 0]  # 420x140
    temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
    [h_pad, w_pad] = temp2.shape
    img_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = img[:, :, i]
        temp2 = np.pad(temp, ((0, 0), (0, 1)), 'symmetric')
        img_pad[:, :, i] = temp2
    print('填充后原图片尺寸:',img_pad.shape)

    # img_pad = img
    # print('不填充图片')

    height, width = img_pad.shape[0], img_pad.shape[1]

    # 原图的缩放
    img_lr = rescale(img_pad, (1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
    # 切patch（多光谱）
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = img_lr[:, :, 0]
    temp_mid = img_pad[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = np.int32(pad_width)
    pad_width_mid = np.floor(patchsize_HSI * 3 / 2)
    pad_width_mid = np.int32(pad_width_mid)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    temp2_mid = np.pad(temp_mid, pad_width_mid-1, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    [h_pad_mid, w_pad_mid] = temp2_mid.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    data_mid_pad = np.empty((h_pad_mid, w_pad_mid, bands), dtype='float32')
    for i in range(bands):
        temp = img_lr[:, :, i]
        temp_mid = img_pad[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        temp2_mid = np.pad(temp_mid, pad_width_mid-1, 'symmetric')
        data_HSI_pad[:, :, i] = temp2
        data_mid_pad[:, :, i] = temp2_mid
    print('二次填充后图片尺寸:',data_mid_pad.shape)
    # 构建高光谱patch序列
    img_list_LRHS=[]
    img_list_mid =[]
    k = 0
    for i in range(int(height/3)):
        for j in range(int(width/3)):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = data_HSI_pad[(i):(i + 2 * pad_width + 1),
                    (j):(j + 2 * pad_width + 1),:]
            patch_mid = data_mid_pad[(i*3):(i*3 + 2 * pad_width_mid + 1),
                    (j*3):(j*3 + 2 * pad_width_mid + 1),:]
            # print(patch.shape,patch_mid.shape)
            img_list_LRHS.append(patch)
            img_list_mid.append(patch_mid)
            k = k + 1
    print('每个高光谱低分辨率图的patch的尺寸：', img_list_LRHS[0].shape)
    print('每个高光谱高分辨率图的patch的尺寸：', img_list_mid[0].shape)

    # 将pad后的原图构建为RGB
    # img_R = np.mean(img_pad[:,:,0:75], axis=2)
    # img_R = mat2gray(img_R)
    # img_G = np.mean(img_pad[:,:,75:150], axis=2)
    # img_G = mat2gray(img_G)
    # img_B = np.mean(img_pad[:,:,150:224], axis=2)
    # img_B = mat2gray(img_B)
    img_R = img_pad[:,:,10]
    img_G = img_pad[:,:,20]
    img_B = img_pad[:,:,35]

    # 切patch（PAN）
    pad_width = np.floor(patchsize_HSI * 3 / 2)
    pad_width = np.int32(pad_width)
    img_R = np.pad(img_R, pad_width-1, 'symmetric')
    img_G = np.pad(img_G, pad_width-1, 'symmetric')
    img_B = np.pad(img_B, pad_width-1, 'symmetric')
    # 构建PAN patch序列
    img_list_PAN=[]
    k = 0
    for i in range(int(height/3)):
        for j in range(int(width/3)):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch1 = img_R[(3*i):(3*i + 2 * pad_width + 1),
                    (3*j):(3*j + 2 * pad_width + 1)]
            patch2 = img_G[(3*i):(3*i + 2 * pad_width + 1),
                    (3*j):(3*j + 2 * pad_width + 1)]
            patch3 = img_B[(3*i):(3*i + 2 * pad_width + 1),
                    (3*j):(3*j + 2 * pad_width + 1)]
            patch = np.stack([patch1,patch2,patch3],axis=2)
            img_list_PAN.append(patch)
            k = k + 1
    print('每个PAN图的patch的尺寸：', img_list_PAN[0].shape)
    # print('每类各', k, '个patch已切完')
    return img_list_LRHS, img_list_PAN, img_list_mid


def create_gt(dir_original_img):
    k = 0
    datas = sio.loadmat(dir_original_img)
    img = np.array(datas['Binary'][...], dtype=np.float32) # 420x140
    img = np.pad(img, ((0, 0), (0, 1)), 'symmetric')
    # img = np.pad(img, ((1, 2), (2, 2)), 'symmetric')
    height,width = img.shape
    # gt_list = torch.zeros(int(height/3)*int(width/3))
    gt_list = []
    for i in range(int(height/3)):
        for j in range(int(width/3)):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            if img[i*3+1,j*3+1] == 1 :
            # if img[i*3,j*3] == 1 :
                #gt_list[k] = 1
                gt_list.append(1)
            elif img[i*3+1,j*3+1] == 0 :
            # elif img[i*3,j*3] == 0 :
                # gt_list[k] = -1
                gt_list.append(0)
            else:
                gt_list.append(2)
            k = k + 1
    high = i+1
    print('读入真实标签数：',k,'图片高度：',high,'图片宽度：',j+1)
    return gt_list,high

def create_pseudo_label(dir,need_numpy=False):
    k = 0
    datas = sio.loadmat(dir)
    img = np.array(datas['result'][...], dtype=np.float32)
    plt.imshow(img)
    plt.suptitle(dir)
    plt.show()
    height,width = img.shape
    label_list = []
    for i in range(int(height)):
        for j in range(int(width)):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            if img[i,j]==1 :
                label_list.append(1)
            elif img[i,j] == 0:
                label_list.append(0)
            else:
                label_list.append(2)
            k = k + 1
    # print('生成伪标签数：',k)
    if need_numpy:
        return label_list,img
    else:
        return label_list

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

def normalize_data_2(data):
    # 获取每个特征的最大值和最小值
    data2 = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    min_vals = np.min(data2, axis=0)
    max_vals = np.max(data2, axis=0)
    # 将数据减去最小值，再除以最大值和最小值的差
    normalized_data2 = (data2 - min_vals) / (max_vals - min_vals)
    normalized_data = normalized_data2.reshape(data.shape[0],data.shape[1],data.shape[2])
    return normalized_data





import os

def create_test_mat_file(file_path, data):
    sio.savemat(file_path, {'T1': data, 'T2': data})


# 测试 create_HS_list 函数
def test_create_HS_list():
    # 创建一个简单的测试数据
    test_data = np.random.rand(420, 140, 224).astype(np.float32)

    # 保存测试数据到 .mat 文件
    test_mat_file = './test_data.mat'
    create_test_mat_file(test_mat_file, test_data)
    PAN_path = '/media/xd132/E1/WHB/xiugao/data/Liyukou/T1_RGB.mat'

    # 调用 create_HS_list 函数
    img_list_LRHS, img_list_PAN, img_list_mid = create_HS_list(test_mat_file, 't1', 3)

    # 打印结果
    print('LRHS Patches:', len(img_list_LRHS))
    print('PAN Patches:', len(img_list_PAN))
    print('Mid Patches:', len(img_list_mid))
    print('First LRHS Patch Shape:', img_list_LRHS[0].shape)
    print('First PAN Patch Shape:', img_list_PAN[0].shape)
    print('First Mid Patch Shape:', img_list_mid[0].shape)

    # 删除测试文件
    # os.remove(test_mat_file)


# 运行测试样例
if __name__ == '__main__':
    test_create_HS_list()