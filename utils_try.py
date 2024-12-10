import torch
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    # lr = start_lr * (0.1 ** (epoch // 20))
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def build_graph_upper(img,device = 'cpu'):
    # img = F.normalize(img,p=2,dim=0) #新增一个归一化
    corr_matrix =  img.transpose(1,0) @ img
    corr_matrix.fill_diagonal_(0)
    x = img.transpose(1, 0)
    _, indices = torch.topk(corr_matrix, 5, dim=1)
    corr_matrix_y = torch.zeros_like(corr_matrix)
    corr_matrix_y = corr_matrix_y.scatter_(1, indices, corr_matrix.gather(1, indices)).detach().cpu()
    corr_matrix = sp.coo_matrix(corr_matrix_y)
    indices,values = from_scipy_sparse_matrix(corr_matrix)
    data = Data(x=x,edge_index=indices,edge_attr=values)
    return data

def build_graph_lower(img,device = 'cpu'):
    corr_matrix_x = img.reshape(img.shape[0],img.shape[1]*img.shape[1],1).squeeze()
    # corr_matrix_x = F.normalize(corr_matrix_x, p=2, dim=0)
    corr_matrix = corr_matrix_x.transpose(1,0) @ corr_matrix_x
    corr_matrix.fill_diagonal_(0)
    x = corr_matrix_x.transpose(1, 0)
    _, indices = torch.topk(corr_matrix, 5, dim=1)
    corr_matrix_y = torch.zeros_like(corr_matrix)
    corr_matrix_y = corr_matrix_y.scatter_(1, indices, corr_matrix.gather(1, indices)).detach().cpu()
    corr_matrix = sp.coo_matrix(corr_matrix_y)
    indices,values = from_scipy_sparse_matrix(corr_matrix)
    data = Data(x=x,edge_index=indices,edge_attr=values)
    return data

def build_graph_middle(img,device = 'cpu'):
    img = F.normalize(img, p=2, dim=1)
    corr_matrix =  img @ img.transpose(1,0)
    # corr_matrix.fill_diagonal_(0)
    # corr_matrix = F.normalize(corr_matrix,p=2,dim=1)
    # corr_matrix = corr_matrix + torch.eye(1).to(device)

    top_k = 5
    if (img.shape[0]<5):
        top_k = img.shape[0] - 1
    _, indices = torch.topk(corr_matrix, top_k, dim=1)
    corr_matrix_y = torch.zeros_like(corr_matrix)
    corr_matrix_y = corr_matrix_y.scatter_(1, indices, corr_matrix.gather(1, indices)).detach().cpu()
    corr_matrix = sp.coo_matrix(corr_matrix_y)
    indices,values = from_scipy_sparse_matrix(corr_matrix)
    data = Data(x=img,edge_index=indices,edge_attr=values)
    return data,corr_matrix_y,top_k

