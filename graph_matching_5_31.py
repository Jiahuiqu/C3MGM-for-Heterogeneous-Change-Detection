import numpy as np
import torch
from graph_matching_utils import unsorted_segment_sum,\
    get_pairwise_similarity,compute_cross_attention
import torch.nn as nn
import graph_matching_utils
import scipy.io as sio
import torch.nn.functional as F

# 上路或下路图匹配
def graph_matching_UPorDOWN(batch_graph, nets, patch_num):
    # encoding层，用GCN来编码图节点信息
    batch_x_up = nets[1](batch_graph.x, batch_graph.edge_index)
    # propagation层，传播，更新节点
    node_new_up = node_update_layer(
        batch_x_up, batch_graph, nets[2], patch_num, nets[3])
    # aggregation层，聚合，得出图向量
    graph_state_up = graph_aggregator(
        node_new_up, batch_graph.batch, patch_num * 2, nets[4])
    t1_graph_state_up, t2_graph_state_up = graph_matching_utils.reshape_and_split_tensor(graph_state_up, 2)
    # 图像量求差，过MLP得预测向量
    # difference_up = t1_graph_state_up - t2_graph_state_up
    difference_up = torch.cat((t1_graph_state_up, t2_graph_state_up),dim=1)
    preds_up = nets[5](difference_up)
    return preds_up,difference_up
def graph_matching_UPorDOWN_new(batch_graph, nets, patch_num, s_diag):
    # encoding层，用GCN来编码图节点信息
    batch_x_up = nets[1](batch_graph.x, batch_graph.edge_index)
    # propagation层，传播，更新节点
    node_new_up = node_update_layer(
        batch_x_up, batch_graph, nets[2], patch_num, nets[3])
    # aggregation层，聚合，得出图向量
    graph_state_up = graph_aggregator(
        node_new_up, batch_graph.batch, patch_num * 2, nets[4])
    t1_graph_state_up, t2_graph_state_up = graph_matching_utils.reshape_and_split_tensor(graph_state_up, 2)
    # 图像量求差，过MLP得预测向量
    # difference_up = t1_graph_state_up - t2_graph_state_up
    difference_up = torch.cat((t1_graph_state_up, t2_graph_state_up), dim=1)
    preds_up,difference_up = nets[5](difference_up, s_diag)
    return preds_up,difference_up


# 中路图匹配
'''
def graph_matching_middle_all(graph1,graph2,A1,A2,net,k,device):

    x_t1_new = net[1](graph1.x, graph1.edge_index)
    x_t2_new = net[1](graph2.x, graph2.edge_index)
    s_diag = x_t1_new - x_t2_new                         # 中路出效果版求法
    s = graph1.x @ graph2.x.transpose(1,0)
    # s = x_t1_new @ x_t2_new.transpose(1,0)
    # G1, H1 = graph_matching_utils.get_G_H(A1,device,k)
    # G2, H2 = graph_matching_utils.get_G_H(A2,device,k)
    # X = graph_matching_utils.reshape_edge_feature(x_t1_new,G1,H1,device)
    # Y = graph_matching_utils.reshape_edge_feature(x_t2_new,G2,H2,device)
    # #
    # # # # x = X.transpose(1,0)
    # # # # x = x.view(-1,5,128)
    # # # # x = x.mean(dim=1)
    # # # # y = Y.transpose(1,0)
    # # # # y = y.view(-1,5,128)
    # # # # y = y.mean(dim=1)
    # # # s_diag = x_t1_new-x_t2_new
    # # # s_diag = graph1.x - graph2.x
    # # # s_diag = net[6](s_diag)
    #
    # Me, Mp = net[2](X, Y, graph1.x, graph2.x)
    # # Me = torch.matmul(X.transpose(1,0),Y)
    # # Mp = torch.matmul(graph1.x, graph2.x.transpose(1, 0))
    # M = torch.diag(Mp.view(-1))
    # N = torch.diag(Me.view(-1))
    # N = torch.kron(G2,G1) @ N @ torch.kron(H2,H1).transpose(0,1)
    # M = M + N
    # v = net[3](M)
    # s = v.view(graph1.x.shape[0], -1).transpose(0, 1)
    # s = torch.nn.functional.normalize(s,dim=0)
    pseudo,minmin,maxmax = gen_pseudo_from_sim(s)
    s_diag_2 = torch.diag(s)
    # s_diag_2 = s_diag_2
    x_t1_new2 = torch.mul(x_t1_new.transpose(1,0), s_diag_2).transpose(0,1)
    x_t2_new2 = torch.mul(x_t2_new.transpose(1,0), s_diag_2).transpose(0,1)
    # s_diag = x_t1_new2 - x_t2_new2
    # s = s.reshape(1,s.shape[0],s.shape[1])
    # s = net[4](s).squeeze()
    # s = s.sub(1/s.shape[0])
    # s = F.normalize(s, p = float('inf'), dim=1)
    # s_diag = torch.diag(s)
    #
    # # if (batch == 6) & ((epoch+1)%10==0) & (type =='train'):
    # #     # print('graph1.x',graph1.x,'\ngraph2.x',graph2.x,'\nMp:',Mp,'\ns:',s)
    # #     x1 = graph1.x.cpu().detach().numpy()
    # #     x2 = graph2.x.cpu().detach().numpy()
    # #     Mpp = Mp.cpu().detach().numpy()
    # #     vv = v.cpu().detach().numpy()
    # #     ss = s.cpu().detach().numpy()
    # #     sio.savemat('./report/batch' + str(batch) + 'epoch' + str(epoch) + '.mat', mdict={'Mp': Mpp,'v':vv,'s':ss,'x1':x1,'x2':x2})
    return s_diag,pseudo,minmin,maxmax'''
def graph_matching_middle_new(graph1, graph2, net):
    # 过GCN
    x_t1_new = net[1](graph1.x, graph1.edge_index) #(64,64)
    x_t2_new = net[1](graph2.x, graph2.edge_index)

    # 128维的差异向量，用来更新伪标签的
    dif_mid = x_t1_new - x_t2_new
    dif_mid = torch.cat((dif_mid, graph1.x-graph2.x), dim=1)

    # 再计算一阶和二阶相关性度量，将CNN和GCN的相关度加起来，得到一阶和二阶的联合相似度，再将维度先分别降下去，再作分类
    c1,c2,g1,g2 = net[4](graph1.x,graph2.x,x_t1_new,x_t2_new)
    s = torch.mm(c1,c2.transpose(1,0)) + torch.mm(g1,g2.transpose(1,0)) #(64,64)
    s_diag = torch.diag(s)
    s_diag = s_diag.unsqueeze(dim=1)
    s_diag = F.softmax(s_diag, dim=0)
    t1 = net[2](x_t1_new)
    t2 = net[2](x_t2_new)
    # t1 = t1 + s_diag * t1
    # t2 = t2 + s_diag * t2
    s_diag = torch.exp(1-s_diag)-1
    t1 = s_diag * t1
    t2 = s_diag * t2
    feature_difference = torch.concat([t1,t2],dim=1)
    preds_mid = net[3](feature_difference)

    return  preds_mid, dif_mid, s_diag


# 中路图匹配
def graph_matching_middle(graph1,graph2,A1,A2,net,k,device):

    x_t1_new = net[1](graph1.x, graph1.edge_index)
    x_t2_new = net[1](graph2.x, graph2.edge_index)
    s_diag = x_t1_new - x_t2_new                         # 中路出效果版求法
    pseudo = None
    minmin = None
    maxmax = None
    return s_diag,pseudo,minmin,maxmax


def gen_pseudo_from_sim(s):
    minmin = 0
    maxmax = 0
    num = s.shape[0]
    pseduo_label = torch.zeros(num)
    # 给s归一化
    # 沿着行的方向计算最小值和最大值
    min_vals, _ = torch.min(s, dim=1, keepdim=True)
    max_vals, _ = torch.max(s, dim=1, keepdim=True)
    # 最小-最大缩放，将x的范围缩放到[0, 1]
    s_1 = (s - min_vals) / (max_vals - min_vals)
    # 沿着列的方向计算最小值和最大值
    min_vals, _ = torch.min(s, dim=0, keepdim=True)
    max_vals, _ = torch.max(s, dim=0, keepdim=True)
    # 最小-最大缩放，将x的范围缩放到[0, 1]
    s_2 = (s - min_vals) / (max_vals - min_vals)

    # 注意，T1和T2要分别都与另外的时刻对应上
    index_min_T1 = torch.argmin(s_1, dim=1)
    index_max_T1 = torch.argmax(s_1, dim=1)
    index_min_T2 = torch.argmin(s_2, dim=0)
    index_max_T2 = torch.argmax(s_2, dim=0)

    # 求方差
    var_T1 = torch.var(s_1, dim=0)
    var_T2 = torch.var(s_2, dim=1)

    # for i in range(num):
    #     if var_T1[i] > 0.1 and var_T2[i] > 0.1:
    #         # 如果这一行中对角线元素是这一行中最小的，则对其赋值为变化的伪标签
    #         if i == index_min_T1[i] and i == index_min_T2[i]:
    #             pseduo_label[i] = 2
    #             pass
    #         # 如果这一行中对角线元素是这一行中最大的，则对其赋值为不变的伪标签
    #         if i == index_max_T1[i] and i == index_max_T2[i]:
    #             pseduo_label[i] = 1
    #             pass
    #         pass
    for i in range(num):
        # 如果这一行中对角线元素是这一行中最小的，则对其赋值为变化的伪标签
        if i == index_min_T1[i] and i == index_min_T2[i]:
            minmin = minmin + 1
            pseduo_label[i] = 2
            pass
        # 如果这一行中对角线元素是这一行中最大的，则对其赋值为不变的伪标签
        if i == index_max_T1[i] and i == index_max_T2[i]:
            maxmax = maxmax + 1
            pseduo_label[i] = 1
            pass
        pass

    return pseduo_label,minmin,maxmax




# 图中的一轮传播（消息传递）
def graph_prop_once(
        x,edge_index,edge_features,message_net):
    #     # x[784*batch_size,128],edge_index[2,729*5*batch_size],edge_features[26618]
    #     # 参数：
    #     # x: [n_nodes, node_state_dim] 浮点张量，节点状态向量，每个节点一行。
    #     # from_idx = edge_index[0], to_idx = edge_index[1]。
    #     # from_idx: [n_edges] int 张量，from 节点的索引。
    #     # to_idx: [n_edges] int 张量，to 节点的索引。
    #     # message_net：将连接的边缘输入映射到消息向量的网络。
    #     # 每个节点的消息。 应该是可调用的，可以像下面这样调用，
    #     # edge_features：如果提供，应该是一个 [n_edges, edge_feature_dim] 浮点张量，每条边的额外特征。 这里是一维向量。
    #     # 返回：
    #     # aggregated_messages：一个 [n_nodes, edge_message_dim] 浮点张量，聚合消息，每个节点一行。
    from_states = x[edge_index[0]]
    to_states = x[edge_index[1]]
    # print('x.shape:',x.shape,'edge_index[0].shape',edge_index[0].shape,'\nfrom_states.shape:',from_states.shape)
    edge_inputs = [from_states,to_states]
    edge_features = edge_features.unsqueeze(1)
    edge_inputs.append(edge_features)
    edge_inputs = torch.cat(edge_inputs,dim=-1)
    # print('edge_inputs.shape:',edge_inputs.shape)
    messages = message_net(edge_inputs)
    tensor = unsorted_segment_sum(messages, edge_index[1], x.shape[0])
    return tensor

# 计算每个节点的聚合消息
def compute_aggregated_messages(
        x,edge_index,edge_features,message_net):
    aggregated_message = graph_prop_once(x,edge_index,edge_features,message_net)
    edge_index_change = edge_index[[1,0]]
    re_aggregated_message = graph_prop_once(x,edge_index_change,edge_features,message_net)
    aggregated_message = aggregated_message + re_aggregated_message
    # aggregated_message = layer_norm1(aggregated_message)
    return aggregated_message

# 计算块对之间的批量注意力。
def batch_block_pair_attention(
        data,block_idx,n_blocks,similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.
    This function partitions the batch data into blocks according to block_idx.
    计算块对之间的批量注意力。
    此函数根据 block_idx 将批处理数据划分为块。

    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
      data: NxD float tensor.
      block_idx: N-dim int tensor.
      n_blocks: integer.
      similarity: a string, the similarity metric.

    Returns:
      attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
      ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)
    results = torch.cat(results, dim=0)

    return results

# 节点更新的计算
def compute_node_update(
        node_states,node_state_inputs,MLP,node_features=None):
    """Compute node updates.计算节点更新
    Args:
      node_states: [n_nodes, node_state_dim] float tensor, the input node
        states.
      node_state_inputs: a list of tensors used to compute node updates.  Each
        element tensor should have shape [n_nodes, feat_dim], where feat_dim can
        be different.  These tensors will be concatenated along the feature
        dimension.
      node_features: extra node features if provided, should be of size
        [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
        different types of skip connections.
    Returns:
      new_node_states: [n_nodes, node_state_dim] float tensor, the new node
        state tensor.
    """
    # 参数：
    # node_states: [n_nodes, node_state_dim] 浮点张量，输入节点状态。
    # node_state_inputs：用于计算节点更新的张量列表。 每个元素张量应具有形状 [n_nodes, feat_dim]，其中 feat_dim 可以不同。 这些张量将沿着特征维度连接起来。
    # node_features：如果提供了额外的节点特征，应该是大小为 [n_nodes, extra_node_feat_dim] 浮点张量，可用于实现不同类型的跳过连接。
    # 返回：
    # new_node_states: [n_nodes, node_state_dim] float tensor，新的节点状态张量。
    node_state_inputs.append(node_states)
    if node_features is not None:
        node_state_inputs.append(node_features)

    if len(node_state_inputs) == 1:
        node_state_inputs = node_state_inputs[0]
    else:
        node_state_inputs = torch.cat(node_state_inputs, dim=-1)
    mlp_output = MLP(node_state_inputs)
    # if self._layer_norm:
    #     mlp_output = nn.self.layer_norm2(mlp_output)
    return mlp_output


# 图传播层，更新节点，交叉图匹配
def node_update_layer(
        batch_x, batch_graph, message_net, batch_size,MLP):
    # 单图信息聚合
    aggregated_message = compute_aggregated_messages(
        batch_x, batch_graph.edge_index, batch_graph.edge_attr, message_net)
    # 求跨图注意力信息
    cross_graph_attention = batch_block_pair_attention(
        batch_x, batch_graph.batch, batch_size * 2, similarity='dotproduct')
    attention_input = batch_x - cross_graph_attention

    return compute_node_update(
        batch_x,[aggregated_message, attention_input],MLP,node_features=batch_graph.x)


# 此模块通过从部分聚合来计算图形表示。
class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    # node_state_dim = 128? graph_rep_dim = 128?
    #         aggregator=dict(
    #             node_hidden_sizes=[graph_rep_dim],
    #             graph_transform_sizes=[graph_rep_dim],
    #             input_size=[node_state_dim],
    #             gated=True,
    #             aggregation_type='sum'),

    def __init__(self,
                 node_hidden_sizes = [128],
                 graph_transform_sizes = [128],
                 input_size = [128],
                 gated = True,
                 aggregation_type = 'sum',
                 name = 'graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        # node_hidden_sizes：节点变换网络的隐藏层大小。 最后一个元素是聚合图形表示的大小。
        # graph_transform_sizes：图形表示之上的转换层的大小。 该列表的最后一个元素是输出图形表示的最终维度。
        #   gated：设置为 True 进行门控聚合，False 则不。
        # aggregation_type: {sum, max, mean, sqrt_n} 之一。

        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()



def graph_aggregator(node_states, graph_idx, n_graphs,MLP2):
    """Compute aggregated graph representations.
    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node states of a
        batch of graphs concatenated together along the first dimension.
      graph_idx: [n_nodes] int tensor, graph ID for each node.
      n_graphs: integer, number of graphs in this batch.
    Returns:
      graph_states: [n_graphs, graph_state_dim] float tensor, graph
        representations, one row for each graph.
    """
    node_states_g = MLP2(node_states)

    # if self._gated: 门控模块？
    #     gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
    #     node_states_g = node_states_g[:, self._graph_state_dim:] * gates
    gates = torch.sigmoid(node_states_g[:, :128]) ###################### 这个self._graph_state_dim不是百分之百确定？貌似是128？
    node_states_g = node_states_g[:, 128:] * gates ########################
    graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

    # if self._aggregation_type == 'max':
    #     # reset everything that's smaller than -1e5 to 0.
    #     graph_states *= torch.FloatTensor(graph_states > -1e5)
    # graph_states *= torch.FloatTensor(graph_states > -1e5)         # 这块后面可能还得看看
    # transform the reduced graph states further

    # if (self._graph_transform_sizes is not None and
    #         len(self._graph_transform_sizes) > 0):
    #     graph_states = self.MLP2(graph_states)

    return graph_states

# 计算成对损失
def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
    """Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """
    # 计算成对损失。
    # 参数：
    # x: [N, D] 浮点张量，代表 N 个例子。
    # y: [N, D] 浮点张量，表示另外 N 个例子。
    # labels：[N] int tensor，取值-1或+1。 labels[i] = +1 如果 x[i] 和 y[i] 相似，否则 -1。
    # loss_type：保证金或汉明。
    # margin: float 标量，margin 为保证金损失。
    # 返回：
    # 损失：[N] 浮动张量。 每对表示的损失。

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - torch.sum((x - y) ** 2, dim=-1)))
        # torch.sum((x - y) ** 2, dim=-1) = euclidean_distance(x, y)
        # This is the squared Euclidean distance.这是平方欧氏距离
    elif loss_type == 'hamming':
        return 0.25 * (labels - torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)) ** 2
        # torch.mean(torch.tanh(x) * torch.tanh(y), dim=1) = approximate_hamming_similarity(x, y)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)

