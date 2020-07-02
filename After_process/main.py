from torch import nn
import torch
import dgl
class GCNs()

class FullConn(nn.Module):
    def __init__(self, mol_feat_size, dgl_feat_size):
        super().__init__()
        self.full_conn_1 = nn.Linear(
            mol_feat_size+dgl_feat_size, mol_feat_size)
        self.full_conn_2 = nn.Linear(mol_feat_size, 1)
        self.activate = nn.Sigmoid()

    def forward(self, feat):
        result_1 = self.full_conn_1(feat)
        result_2 = self.full_conn_2(result_1)
        result_end = self.activate(result_2.reshape(-1))
        return result_end


class FeatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dgl_feat, mol_feat):
        return torch.cat((dgl_feat, mol_feat), -1).reshape(1, -1)


class ReadOut(nn.Module):
    def __init__(self, input_atom_feat_size, hidden_feat_size):
        super().__init__()
        self.full_conn = nn.Linear(
            input_atom_feat_size+hidden_feat_size, hidden_feat_size)
        self.activate = nn.Sigmoid()

    def readout_msg(self, edge):
        edge_data = edge.data['hidden']
        return {'edge_mail': edge_data}

    def readout_reduce(self, node):
        node_mail = torch.mean(node.mailbox['edge_mail'], 1)
        # TODO sum 改成了 mean
        return {'out_sum': node_mail}

    def readout_nodeupdate(self, node):
        node_origin_feat = node.data['n_input']
        node_sum_feat = node.data['out_sum']
        node_feat_cat = torch.cat((node_origin_feat, node_sum_feat), -1)
        node_feat_mm = self.full_conn(node_feat_cat)
        node_feat_act = self.activate(node_feat_mm)
        node_feat_final = node_feat_act
        return {'feat_final': node_feat_final}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.update_all(
            self.readout_msg, self.readout_reduce, self.readout_nodeupdate)
        gcn_feat = dgl_data.ndata['feat_final']
        result = torch.mean(gcn_feat, 0)
        # TODO sum 改成 mean
        return result


class FeatConvert(nn.Module):
    def __init__(self, input_atom_feat_size, input_edge_feat_size, hidden_feat_size):
        super().__init__()
        self.full_conn_a = nn.Linear(input_atom_feat_size, hidden_feat_size)
        self.full_conn_e = nn.Linear(input_edge_feat_size, hidden_feat_size)
        self.activate = nn.Sigmoid()

    def forward(self, dgl_data: dgl.DGLGraph):
        node_feature = dgl_data.ndata['n_input']
        node_feature_new = self.full_conn_a(node_feature)
        edge_feature = dgl_data.edata['e_input']
        edge_feature_new = self.full_conn_e(edge_feature)
        node_feature_act = self.activate(node_feature_new)
        edge_feature_act = self.activate(edge_feature_new)
        dgl_data.ndata['n_input'] = node_feature_act
        dgl_data.edata['e_input'] = edge_feature_act
        return dgl_data


class EdgeFeatInit(nn.Module):
    def __init__(self, input_atom_feat_size, input_edge_feat_size, hidden_feat_size):
        super().__init__()
        self.full_conn = nn.Linear(
            input_atom_feat_size+input_edge_feat_size, hidden_feat_size)
        self.activate = nn.Sigmoid()

    def edge_init(self, edge):
        edge_data = edge.data['e_input']
        edge_src = edge.src['n_input']
        edge_concate = torch.cat((edge_data, edge_src), -1)
        edge_init = self.full_conn(edge_concate)
        edge_act = self.activate(edge_init)
        return {'init': edge_act, 'hidden': edge_act}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.apply_edges(self.edge_init)
        return dgl_data


class SingleLayer(nn.Module):
    def __init__(self, hidden_feat_size):
        super().__init__()
        self.full_conn = nn.Linear(hidden_feat_size, hidden_feat_size)
        self.activate = nn.Sigmoid()

    def gcn_msg(self, edge):  # 结点到边的信息传递（通过src和dst获取源点和目标点的特征）
        # 将所有的边，每一个维度的数据整合在一起，比如100条边，每条边有长度为8的向量，那么转换后就成了10*100的张量
        edge_data = edge.data['hidden']
        return {'edge_mail': edge_data}

    def gcn_reduce(self, node):  # 边到结点的汇聚（通过mailbox函数获取结点的邻边的所有信息）
        # 选取n个结点*feature，选择多少个结点是临时决定的（由计算量决定）
        node_mail = torch.mean(node.mailbox['edge_mail'], 1)  # 代表把边进行汇合
        # TODO sum 改成了 mean
        return {'mail': node_mail}

    def edge_update(self, edge):
        pseud_converge = edge.src['mail']-edge.data['hidden']
        feature_add_init = torch.add(pseud_converge, edge.data['init'])
        feature_mm = self.full_conn(feature_add_init)
        feature_act = self.activate(feature_mm)
        return {'hidden': feature_act}

    def forward(self, dgl_data: dgl.DGLGraph):
        # 注意这个函数只会更新node feature
        dgl_data.update_all(self.gcn_msg, self.gcn_reduce)
        dgl_data.apply_edges(self.edge_update)
        return dgl_data


class DMPNN(nn.Module):
    def __init__(self, hidden_feat_size, hidden_layer_num):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for index in range(hidden_layer_num):
            self.hidden_layers.append(SingleLayer(hidden_feat_size))

    def forward(self, dgl_data):
        for layer in self.hidden_layers:
            dgl_data = layer(dgl_data)
        return dgl_data


class MainModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_atom_feat_size = args.atom_feat_size
        input_edge_feat_size = args.bond_feat_size
        hidden_feat_size = args.hidden_feat_size
        mol_feat_size = args.mol_feat_size
        hidden_layer_num = args.hidden_layer_num
        self.convert = FeatConvert(
            input_atom_feat_size, input_edge_feat_size, hidden_feat_size)
        self.edge_init = EdgeFeatInit(
            input_atom_feat_size, input_edge_feat_size, hidden_feat_size)
        self.dmpnn = DMPNN(hidden_feat_size, hidden_layer_num)
        self.graph_reader = ReadOut(input_atom_feat_size, hidden_feat_size)
        self.feat_fusion = FeatFusion()
        self.full_conn = FullConn(mol_feat_size, hidden_feat_size)

    def forward(self, dgl_data, mol_feat):
        '''
        dgl_after_conv = self.convert(dgl_data)
        dgl_after_gcn = self.dmpnn(dgl_after_conv)
        '''
        dgl_after_init = self.edge_init(dgl_data)
        dgl_after_gcn = self.dmpnn(dgl_after_init)
        dgl_feat = self.graph_reader(dgl_after_gcn)
        fusion_feat = self.feat_fusion(dgl_feat, mol_feat)
        result = self.full_conn(fusion_feat)
        return result


if __name__ == '__main__':
    pass
