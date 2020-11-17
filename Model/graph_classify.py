from torch import nn
import torch
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        '''
        定义图神经分类模型的结构
        :param in_dim: 输入的特征维数， 与forward中的第一层的输入特征数相对应
        :param hidden_dim: 隐藏层单元数
        :param n_classes: 分类数
        '''
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        '''
        向前传播
        :param g: 图
        :return:
        '''
        h = g.in_degrees().view(-1, 1).float()  # 使用节点的度作为节点初始特征， 对于无向图来说，输入度等于输出度
        # 卷积层以及激活函数
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['feat'] = h
        # 以平均值来代表图
        hg = dgl.mean_nodes(g, 'feat')
        return self.classify(hg)


class DGLInit(nn.Module):
    def __init__(self, innode_size, inedge_size, out_size):
        super().__init__()
        self.init_weight_node = nn.Linear(innode_size, out_size, bias=True)
        self.init_weight_edge = nn.Linear(inedge_size, out_size, bias=True)

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.ndata['resizedfeature'] = self.init_weight_node(
            dgl_data.ndata['feat'])
        dgl_data.ndata['hidden'] = dgl_data.ndata['resizedfeature']
        dgl_data.ndata['stack'] = dgl_data.ndata['resizedfeature']

        dgl_data.edata['resizedfeature'] = self.init_weight_edge(
            dgl_data.edata['feat'])
        dgl_data.edata['hidden'] = dgl_data.edata['resizedfeature']
        dgl_data.edata['stack'] = dgl_data.edata['resizedfeature']
        return dgl_data


class SingleGCN(nn.Module):
    def __init__(self, in_feats, out_feats, weight, acti):
        super().__init__()
        self.gcn_weight = weight
        self.GCNlayers = acti

    def msg_gcn(self, edge):
        msg = torch.div(edge.src['hidden'], edge.src['degree'])
        return {'msg': msg}

    def reduce_gcn(self, node):
        reduce = torch.sum(node.mailbox['msg'], 1)
        reduce = reduce+node.data['hidden']
        return {'reduce': reduce}

    def apply_gcn(self, node):
        data = node.data['reduce']
        result = self.gcn_weight(data)
        return {'hidden': result}

    def forward(self, dgl_data: dgl.DGLGraph):
        # stack在此处只有记录的作用
        dgl_data.update_all(self.msg_gcn, self.reduce_gcn, self.apply_gcn)
        dgl_data.ndata['stack'] = torch.cat(
            [dgl_data.ndata['stack'], dgl_data.ndata['hidden']], 1)
        return dgl_data


class GCNProcess(nn.Module):
    def __init__(self, size, layer):
        super().__init__()
        self.acti = nn.ReLU()
        self.gcn_weight = nn.Linear(size, size, bias=True)
        self.GCNlayers = nn.ModuleList()
        for lay in range(layer):
            self.GCNlayers.append(
                SingleGCN(size, size, self.gcn_weight, self.acti))

    def forward(self, dgl_data):
        for model in self.GCNlayers:
            dgl_data = model(dgl_data)
        return dgl_data


class GCNPredict(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dgl_data):
        dgl_predict = torch.mean(dgl_data.ndata['hidden'], 0).reshape(1, -1)
        return dgl_predict


class Predictwithbase(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.predict = nn.Linear(in_size, out_size)
        # self.soft = nn.Softmax(-1)

    def forward(self, dgl_feat, base_feat):
        final_feat = torch.cat([dgl_feat, base_feat], -1)
        result = self.predict(final_feat)
        # result = self.soft(result)
        return result


class Predictnobase(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.predict = nn.Linear(in_size, out_size)
        # self.soft = nn.Softmax(-1)

    def forward(self, dgl_feat):
        result = self.predict(dgl_feat)
        # result = self.soft(result)
        return result


class PredictOnlyBase(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.predict = nn.Linear(in_size, out_size)
        # self.soft = nn.Softmax(-1)

    def forward(self, base_feat):
        result = self.predict(base_feat)
        # result = self.soft(result)
        return result


class Node_feat_fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def msg_gcn(self, edge):
        msg = edge.src['hidden']
        return {'msg': msg}

    def reduce_gcn(self, node):
        reduce = torch.sum(node.mailbox['msg'], 1)
        return {'reduce': reduce}

    def apply_gcn(self, node):
        data = node.data['reduce']
        return {'hidden': data}

    def forward(self, dgl_data: dgl.DGLGraph):
        # while len(dgl_data.nodes) > 1:
        #     leftedNodeNum = max(len(dgl_data.nodes)//2, 1)

        dgl_data.update_all(self.msg_gcn, self.reduce_gcn, self.apply_gcn)
        return dgl_data


class SimpleModel(nn.Module):
    def __init__(self, nodefeatsize, edgefeatsize, graphfeatsize, hidden_size, gcn_layers, classnum):
        super().__init__()
        self.nodeedge_feat_init = DGLInit(
            nodefeatsize, edgefeatsize, hidden_size)
        self.graph_feat_init = nn.Linear(graphfeatsize, hidden_size)
        self.gcn_process = GCNProcess(hidden_size, gcn_layers)
        self.gcn_predict = GCNPredict()
        self.predictwithbase = Predictwithbase(hidden_size*2, classnum)
        self.predictnobase = Predictnobase(hidden_size, classnum)
        self.predictonlybase = PredictOnlyBase(hidden_size, classnum)
        self.edge2node_feat = Node_feat_fusion()

    def forward(self, dgl_data, base_data):
        base_feat = self.graph_feat_init(base_data)

        # 将edge特征替代node特征，初始的情况就是简单的相加，忽略原来的node
        dgl_digit = self.nodeedge_feat_init(dgl_data)
        nodedata_init = dgl_digit.ndata['hidden']
        dgl_digit = self.edge2node_feat(dgl_digit)
        nodedata_from_edge = dgl_digit.ndata['hidden']
        dgl_digit = self.gcn_process(dgl_digit)
        nodedata_aftergcn = dgl_digit.ndata['hidden']
        dgl_feat = self.gcn_predict(dgl_digit)
        # predict = self.predictwithbase(dgl_feat, base_feat)
        predict = self.predictonlybase(base_feat)
        # predict = self.predictnobase(dgl_feat)
        return predict


if __name__ == '__main__':
    pass
