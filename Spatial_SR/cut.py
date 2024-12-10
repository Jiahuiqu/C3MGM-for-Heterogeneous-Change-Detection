import numpy as np
import scipy.io as sio
import scipy
from skimage.transform import rescale
import torch
import time


def create_train_data(t1_img):
    print('===> start create train data')
    time1 = time.time()

    data_HSI = t1_img
    patchsize_HSI = 9

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
    # TrainPatch_gt = np.empty((TrainNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32') # 9x9
    # TrainPatch_ms = np.empty((TrainNum, bands, int(patchsize_HSI/3), int(patchsize_HSI/3)), dtype='float32') # 3x3
    # TrainPatch_lms = np.empty((TrainNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32') # 9x9
    TrainPatch_gt = np.empty((TrainNum, patchsize_HSI, patchsize_HSI, bands), dtype='float32') # 9x9
    TrainPatch_ms = np.empty((TrainNum, int(patchsize_HSI/3), int(patchsize_HSI/3), bands), dtype='float32') # 3x3
    TrainPatch_lms = np.empty((TrainNum, patchsize_HSI, patchsize_HSI, bands), dtype='float32') # 9x9

    print('shape of the output:<gt,ms,lms> :<',TrainPatch_gt.shape,TrainPatch_ms.shape,TrainPatch_lms.shape,'>')
    # TrainLabel_HSI = np.empty(TrainNum)

    k = 0

    for i in range(height):
        for j in range(width):
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = data_HSI_pad[(i):(i + 2 * pad_width + 1),
                    (j):(j + 2 * pad_width + 1), :]

            # patch_ms = patch.resize((3,3), resample = Image.BICUBIC)
            # patch_ms = cv2.resize(patch, (3, 3), interpolation=cv2.INTER_CUBIC)
            # print(patch_ms.shape)


            # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
            # patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
            # patch = np.transpose(patch)
            # patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
            TrainPatch_gt[k, :, :, :] = patch
            # patch_ms =rescale(patch, (1,1/3,1/3), order=3, mode='edge', anti_aliasing=True)
            patch_ms = rescale(patch, ( 1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
            # rescale 函数靠谱吗？不确定，还是matlab的imresize好用，貌似
            TrainPatch_ms[k, :, :, :] = patch_ms
            patch_lms = rescale(patch_ms, (3, 3, 1), order=3, mode='edge', anti_aliasing=True)

            TrainPatch_lms[k, :, :, :] = patch_lms

            scipy.io.savemat('cut_data/data_here2/trains/train_data_' + str(k) + '.mat',
                             mdict={'gt': patch, 'ms': patch_ms, 'lms': patch_lms,'num':k})
            if ((k-1)%1000==0):
                print(k-1, '个patch已切,共有',TrainNum,'个')
                time1 = time.time() - time1
                print('生成1000个patch耗时{:.2f}秒', time1)
                time1 = time.time()
            k = k + 1
    print('总计',k,'个patch已切完,successfully created train_data')


def create_test_data(t1_img):
    print('===> start create train data')
    time1 = time.time()

    data_HSI = t1_img
    patchsize_HSI = 27

    height, width, bands = data_HSI.shape
    # ##### 给HSI和LiDAR打padding #####
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = data_HSI[:, :, 0]   #141x48
    pad_width = np.floor(patchsize_HSI / 2) #27/2 = 13
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

    # #### 构建高光谱测试集 ####
    TrainNum = width*height
    TrainPatch_gt = np.empty((TrainNum, patchsize_HSI, patchsize_HSI, bands), dtype='float32') # 27x27
    TrainPatch_ms = np.empty((TrainNum, int(patchsize_HSI/3), int(patchsize_HSI/3), bands), dtype='float32') # 9x9
    TrainPatch_lms = np.empty((TrainNum, patchsize_HSI, patchsize_HSI, bands), dtype='float32') # 27x27

    print('shape of the test_img output:<gt,ms,lms> :<',TrainPatch_gt.shape,TrainPatch_ms.shape,TrainPatch_lms.shape,'>')
    # TrainLabel_HSI = np.empty(TrainNum)

    k = 0

    for iii in range(int(height/8)-2):
        for jjj in range(int(width/8)-2):
            i = 6 * iii + 17
            j = 6 * jjj + 17
            # x是打了padding的高光谱
            # 取第i个训练patch，取一个立方体
            patch = data_HSI_pad[(i):(i + 2 * pad_width + 1),
                    (j):(j + 2 * pad_width + 1), :]
            # print(patch.shape)
            TrainPatch_gt[k, :, :, :] = patch
            # patch_ms =rescale(patch, (1,1/3,1/3), order=3, mode='edge', anti_aliasing=True)
            patch_ms = rescale(patch, ( 1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
            # rescale 函数靠谱吗？不确定，还是matlab的imresize好用，貌似
            TrainPatch_ms[k, :, :, :] = patch_ms
            patch_lms = rescale(patch_ms, (3, 3, 1), order=3, mode='edge', anti_aliasing=True)

            TrainPatch_lms[k, :, :, :] = patch_lms

            scipy.io.savemat('cut_data/data_here2/tests/test_data_' + str(k) + '.mat',
                             mdict={'gt': patch, 'ms': patch_ms, 'lms': patch_lms,'num':k})
            k = k + 1
    time1 = time.time() - time1
    print('生成测试样本耗时{:.2f}秒',time1)
    print('总计',k,'个测试样本,successfully created test_data')


def create_org_img(load_dir):
    # 生成真值为423x144，ms为141x48的original_img
    datas = sio.loadmat(load_dir)
    t1_img = np.array(datas['T1'][...], dtype=np.float32)
    print('orl_img shape :', t1_img.shape)
    height, width, bands = t1_img.shape
    temp = t1_img[:, :, 0]  # 420x140
    temp2 = np.pad(temp, ((1, 2), (2, 2)), 'symmetric')
    [h_pad, w_pad] = temp2.shape
    print('img shape after padding:', temp2.shape)
    t1_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = t1_img[:, :, i]
        temp2 = np.pad(temp, ((1, 2), (2, 2)), 'symmetric')
        t1_pad[:, :, i] = temp2
    ms = rescale(t1_pad, (1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
    lms = rescale(ms, (3, 3, 1), order=3, mode='edge', anti_aliasing=True)
    print('shape of the orl_img:<gt,ms,lms> :<', t1_pad.shape, ms.shape, lms.shape, '>')
    scipy.io.savemat('cut_data/data_here2/origin/orl_img' + '.mat', mdict={'gt': t1_pad, 'ms': ms, 'lms': lms})
    print('successfully created original_img')

    #生成真值为真值为141x48，ms为47x16的eval_img
    ms_new = rescale(ms, (1 / 3, 1 / 3, 1), order=3, mode='edge', anti_aliasing=True)
    lms_new = rescale(ms_new, (3, 3, 1), order=3, mode='edge', anti_aliasing=True)
    print('shape of the eval_img:<gt,ms,lms> :<', ms.shape, ms_new.shape, lms_new.shape, '>')
    scipy.io.savemat('cut_data/data_here2/evals/eval_img' + '.mat', mdict={'gt': ms, 'ms': ms_new, 'lms': lms_new})
    print('successfully created eval_img')
    return ms,t1_pad



def cut_data(load_dir,need_create_testdata = False):
    img_lr,img_hr = create_org_img(load_dir)
    print('用于生成训练样本的低分高光图尺寸：',img_lr.shape)
    print('用于生成测试样本的原图尺寸：',img_hr.shape)
    create_train_data(img_lr)
    if (need_create_testdata == True):
        create_test_data(img_hr)






    #### 数据转换以及把数据搬到GPU #####
    device = torch.device('cuda')
    TrainPatch_gt = torch.from_numpy(TrainPatch_gt).to(device)

load_dir = '/run/media/xd132/E/YHD/X师姐的网/SSPSR-master/dataset/China.mat'
# cut_data(load_dir,need_create_testdata = False)

