import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import Tensor
import pygmtools as pygm


# 下支路
class CNN(nn.Module):
    def __init__(self,net_size1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 256, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, net_size1, kernel_size=(3,3), padding=1)
        self.bn4 = nn.BatchNorm2d(net_size1)

    def forward(self, x, pool):
        # x = x[:, :, 9:-9, 9:-9]
        x = x[:, :, 7:-7, 7:-7]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, net_size1, net_size2 = 128):
        super(GCN, self).__init__()
        # 定义两层 GraphConv 层
        self.conv1 = GCNConv(net_size1,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, net_size2)

    def forward(self, x, edge_index):
        # x 是节点特征矩阵，edge_index 是边索引矩阵
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        return x

class message_net(torch.nn.Module):
    def __init__(self):
        super(message_net, self).__init__()
        self.linear1 = torch.nn.Linear(128*2+1,128*2)
        self.linear2 = torch.nn.Linear(128*2,128*2)
        # self.norm1 = torch.nn.LayerNorm()

    def forward(self,edge_inputs):
        edge_inputs = F.relu(self.linear1(edge_inputs))
        edge_inputs = self.linear2(edge_inputs)
        # edge_inputs = self.norm1(edge_inputs)
        return edge_inputs

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(128*4+64,128*2)
        self.linear2 = nn.Linear(128*2,128)

    def forward(self,node_state_inputs):
        node_state_inputs = F.relu(self.linear1(node_state_inputs))
        node_state_inputs = self.linear2(node_state_inputs)
        return node_state_inputs

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.linear1 = nn.Linear(128,128*2)
        self.linear2 = nn.Linear(128*2,128*2)

    def forward(self,node_state_inputs):
        node_state_inputs = F.relu(self.linear1(node_state_inputs))
        node_state_inputs = self.linear2(node_state_inputs)
        return node_state_inputs

class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.linear1 = nn.Linear(128*2,64)
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(64,8)
        self.linear3 = nn.Linear(8,2)

    def forward(self,node_state_inputs,s_diag=None):
        difference = F.relu(self.linear1(node_state_inputs))
        # node_state_inputs = self.dropout1(node_state_inputs)
        node_state_inputs = F.relu(self.linear2(difference))
        if s_diag != None:
            node_state_inputs = node_state_inputs + s_diag * node_state_inputs
        node_state_inputs = self.linear3(node_state_inputs)
        return node_state_inputs,difference


# 上支路
class MLP0(nn.Module):
    def __init__(self,channels,net_size1,net_size2):
        super(MLP0, self).__init__()
        # self.linear1 = nn.Linear(channels,net_size2)
        # self.linear2 = nn.Linear(net_size2,net_size1)
        self.linear3 = nn.Linear(channels,net_size1)

    def forward(self,node_state_inputs):
        # node_state_inputs = F.relu(self.linear1(node_state_inputs))
        # node_state_inputs = self.linear2(node_state_inputs)
        node_state_inputs = self.linear3(node_state_inputs)
        return node_state_inputs

class GCNup(torch.nn.Module):
    def __init__(self,patchsize):
        super(GCNup, self).__init__()
        # 定义两层 GraphConv 层
        # self.conv1 = GCNConv(49, 128)                      # 4.13改，这块好像只能用硬标签
        self.conv1 = GCNConv(patchsize * patchsize, 128)     # 4.17又改回来，带上超分不太好弄
        self.bn1 = nn.BatchNorm1d(128) # 4.12新增
        self.conv2 = GCNConv(128, 128)

    def forward(self, x, edge_index):
        # x 是节点特征矩阵，edge_index 是边索引矩阵
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        return x

class message_netup(torch.nn.Module):
    def __init__(self):
        super(message_netup, self).__init__()
        self.linear1 = torch.nn.Linear(128*2+1,128*2)
        self.linear2 = torch.nn.Linear(128*2,128*2)
        # self.norm1 = torch.nn.LayerNorm()

    def forward(self,edge_inputs):
        edge_inputs = F.relu(self.linear1(edge_inputs))
        edge_inputs = self.linear2(edge_inputs)
        # edge_inputs = self.norm1(edge_inputs)
        return edge_inputs

class MLPup(nn.Module):
    def __init__(self):
        super(MLPup, self).__init__()
        # self.linear1 = nn.Linear(128*4+49,128*2)   # 4.13改，这块好像只能用硬标签
        self.linear1 = nn.Linear(128*4+81,128*2)     # 4.17改回
        self.linear2 = nn.Linear(128*2,128)

    def forward(self,node_state_inputs):
        node_state_inputs = F.relu(self.linear1(node_state_inputs))
        node_state_inputs = self.linear2(node_state_inputs)
        return node_state_inputs

class MLP2up(nn.Module):
    def __init__(self):
        super(MLP2up, self).__init__()
        self.linear1 = nn.Linear(128,128*2)
        self.linear2 = nn.Linear(128*2,128*2)

    def forward(self,node_state_inputs):
        node_state_inputs = F.relu(self.linear1(node_state_inputs))
        node_state_inputs = self.linear2(node_state_inputs)
        return node_state_inputs

# 中路
class CNNmid(nn.Module):
    def __init__(self,channel,patchsize,down_size,out_size,hidden_size = 32):
        super(CNNmid, self).__init__()
        self.conv_temp_1 = nn.Conv2d(in_channels=channel, out_channels=256, kernel_size=(3, 3),padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.pooling_temp_1 = nn.AvgPool2d(kernel_size=2)
        self.conv_temp_2 = nn.Conv2d(256, 128, (3, 3),padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pooling_temp_2 = nn.AvgPool2d(2)
        self.conv_temp_3 = nn.Conv2d(128, 64, (3, 3),padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling_temp_3 = nn.AvgPool2d(2)

        # self.up_fc1 = nn.Linear(patchsize * patchsize, hidden_size)
        # self.up_fc2 = nn.Linear(patchsize * patchsize, hidden_size)
        # self.down_fc1 = nn.Linear(down_size, hidden_size)
        # self.down_fc2 = nn.Linear(down_size, hidden_size)
        self.up_fc1 = nn.Linear(patchsize * patchsize, hidden_size)
        self.up_fc11 = nn.Linear(hidden_size, hidden_size)
        self.up_fc2 = nn.Linear(patchsize * patchsize, hidden_size)
        self.up_fc22 = nn.Linear(hidden_size, hidden_size)
        self.down_fc1 = nn.Linear(down_size, hidden_size)
        self.down_fc11 = nn.Linear(hidden_size, hidden_size)
        self.down_fc2 = nn.Linear(down_size, hidden_size)
        self.down_fc22 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_up, attention_down, need_attention=True):
        x = x[:, :, 10:-11, 10:-11]

        if need_attention:
            attention_down1 = self.down_fc1(attention_down.reshape(-1, attention_down.shape[2]))  # 64x36,32
            attention_down1 = self.down_fc11(attention_down1)
            attention_down2 = self.down_fc2(attention_down.reshape(-1, attention_down.shape[2]))
            attention_down2 = self.down_fc22(attention_down2)
            attention_down1 = attention_down1.reshape(attention_down.shape[0], attention_down.shape[1], -1)  # 64,36,32
            attention_down2 = attention_down2.reshape(attention_down.shape[0], attention_down.shape[1], -1)
            attention_down2 = attention_down2.permute(0, 2, 1)  # 64,32,36
            attention_down = torch.bmm(attention_down1, attention_down2)  # 64,36,36

            x = x.permute(0, 2, 3, 1)  # 64.6.6.224
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # 64.36.224
            x = torch.bmm(attention_down, x)  # 64.36.224
            x = x.reshape(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), x.shape[2])  # 64.6.6.224
            x = x.permute(0, 3, 1, 2)  # 64.224.6.6

        x = F.relu(self.conv_temp_1(x))  # 64.256.6.6
        x = self.bn1(x)
        x = F.relu((self.conv_temp_2(x)))  # 64.128.6.6
        x = self.bn2(x)
        x = self.pooling_temp_2(x)  # 64.128.3.3
        x = F.relu((self.conv_temp_3(x)))
        x = self.bn3(x)
        x = self.pooling_temp_3(x)
        x = x.view(x.size(0), x.size(1), -1)

        if need_attention:
            # 上路注意力
            attention_up1 = self.up_fc1(attention_up.reshape(-1, attention_up.shape[2]))
            attention_up1 = self.up_fc11(attention_up1)
            attention_up2 = self.up_fc2(attention_up.reshape(-1, attention_up.shape[2]))
            attention_up2 = self.up_fc22(attention_up2)
            attention_up1 = attention_up1.reshape(attention_up.shape[0], attention_up.shape[1], -1)
            attention_up2 = attention_up2.reshape(attention_up.shape[0], attention_up.shape[1], -1)
            attention_up2 = attention_up2.permute(0, 2, 1)
            attention_up = torch.bmm(attention_up1, attention_up2)
            x = torch.bmm(attention_up, x)
        return x.squeeze()

class GCNmid(torch.nn.Module):
    def __init__(self, net_size1=64, net_size2 = 64):
        super(GCNmid, self).__init__()
        # 定义两层 GraphConv 层
        self.conv1 = GCNConv(net_size1,net_size2)
        self.bn1 = nn.BatchNorm1d(net_size2)

    def forward(self, x, edge_index):
        # x 是节点特征矩阵，edge_index 是边索引矩阵
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        return x

class linear_difference(nn.Module):
    def __init__(self):
        super(linear_difference, self).__init__()
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 8)

    def forward(self, feature):
        feature = F.relu(self.linear1(feature))
        feature = F.relu(self.linear2(feature))
        feature = F.relu(self.linear3(feature))
        return feature

class linear_classification(nn.Module):
    def __init__(self):
        super(linear_classification, self).__init__()
        self.linear = nn.Linear(16, 2)

    def forward(self, feature_difference):
        pred = self.linear(feature_difference)
        return pred

class linear_attention(nn.Module):
    def __init__(self):
        super(linear_attention, self).__init__()
        # 定义两个线性层，求注意力权重
        self.linear_CNN_1 = nn.Linear(64, 64)
        self.linear_CNN_2 = nn.Linear(64, 64)
        self.bn_CNN_1 = nn.BatchNorm1d(64)
        self.bn_CNN_2 = nn.BatchNorm1d(64)

        self.linear_GCN_1 = nn.Linear(64, 64)
        self.linear_GCN_2 = nn.Linear(64, 64)
        self.bn_GCN_1 = nn.BatchNorm1d(64)
        self.bn_GCN_2 = nn.BatchNorm1d(64)

    def forward(self, CNN_t1, CNN_t2, GCN_t1, GCN_t2):
        # 因为之前CNN的输出没有归一化，因此先过BN
        CNN_t1_4_sim = self.bn_CNN_1(CNN_t1)
        CNN_t1_4_sim = self.linear_CNN_1(CNN_t1_4_sim)
        CNN_t2_4_sim = self.bn_CNN_2(CNN_t2)
        CNN_t2_4_sim = self.linear_CNN_2(CNN_t2_4_sim)
        GCN_t1_4_sim = self.bn_GCN_1(GCN_t1)
        GCN_t1_4_sim = self.linear_GCN_1(GCN_t1_4_sim)
        GCN_t2_4_sim = self.bn_GCN_2(GCN_t2)
        GCN_t2_4_sim = self.linear_GCN_2(GCN_t2_4_sim)
        return CNN_t1_4_sim,CNN_t2_4_sim,GCN_t1_4_sim,GCN_t2_4_sim
# 328->1
class NetsExtra(nn.Module):
    def __init__(self):
        super(NetsExtra, self).__init__()
        self.linear1 = nn.Linear(128 * 2, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32,8)
        self.linear4 = nn.Linear(8,1)

    def forward(self,node_state_inputs):
        node_state_inputs = F.relu(self.linear1(node_state_inputs))
        node_state_inputs = self.dropout1(node_state_inputs)
        node_state_inputs = F.relu(self.linear2(node_state_inputs))
        node_state_inputs = F.relu(self.linear3(node_state_inputs))
        node_state_inputs = self.linear4(node_state_inputs)
        return node_state_inputs.squeeze()
