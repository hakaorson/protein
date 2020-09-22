import os
import re
import dgl
from dgl import subgraph
import numpy as np
import networkx as nx
import torch
import random


def read_file(path):  # 读取文件的常规方法
    result = []
    with open(path, 'r') as f:
        for line in f:
            data = re.split(' |\t', line.strip())
            result.append(tuple(data))
    return result


class DataLoaderGraph():  # 对蛋白质网络里面的dip进行研究
    def __init__(self, graph_name, graph_path, embedding_path):
        self.graph_path = graph_path
        self.embedding_path = embedding_path
        self.graph_name = graph_name
        self.origin_node_embed = self.load_embedding('node')  # dict 类型——结点——数据
        self.origin_edge_embed = self.load_embedding('edge')  # dict 类型——边——数据
        self.edges = self.get_edgelist()  # list 边
        self.nodes = self.get_nodelist()  # list 结点

    def load_embedding(self, nodeoredge):  # embedding可以尽可能多
        path = os.path.join(self.embedding_path, nodeoredge)
        embedding_types = os.listdir(path)
        result = {}
        for embedding_type in embedding_types:
            type_path = os.path.join(path, embedding_type)
            datas = read_file(type_path)
            embedding = {}
            for data in datas:
                if nodeoredge == 'node':
                    embedding[data[0]] = list(map(float, data[1:]))
                elif nodeoredge == 'edge':
                    embedding[tuple(data[:2])] = list(map(float, data[2:]))
                else:
                    pass
            result[embedding_type] = embedding
        return result

    def get_edgelist(self):
        edgelist = read_file(os.path.join(self.graph_path))
        return [tuple(item) for item in edgelist]

    def get_nodelist(self):
        nodeset = set()
        for item in self.edges:
            nodeset.add(item[0])
            nodeset.add(item[1])
        return list(nodeset)


class DictData():
    def __init__(self, dict_data):
        self.dict_data = dict_data
        self.all_data = self.extract_from_dict()  # np array
        # 对于每一种embeding，都可以设置默认值
        self.default = list(np.mean(self.all_data, 0))

    def extract_from_dict(self):
        result = []
        for item in self.dict_data.keys():
            temp_data = self.dict_data[item]
            result.append(temp_data)
        return np.array(result)


class DataLoaderFiles():
    def __init__(self, path):
        self.path = path
        self.dict_datas = self.get_datas()
        self.merge_datas = self.merge_datas()

    def get_datas(self):
        result = {}
        if os.path.isdir(self.path):
            names = os.listdir(self.path)
            for name in names:
                temp_path = os.path.join(self.path, name)
                result[name] = read_file(temp_path)
            return result
        else:
            result['all'] = read_file(self.path)
            return result

    def merge_datas(self):
        result = []
        for typ in self.dict_datas.keys():
            temp_data = self.dict_datas[typ]
            result.extend(temp_data)
        return list(set(result))


class FeatureEngineer():
    def __init__(self, basedata: DataLoaderGraph):
        self.basedata = basedata
        self.node_input_feature = self.feature_process_node()  # 节点——类型——数据
        self.node_merge_feature = self.feature_merge_node()  # 节点——数据
        self.node_defau_feature = DictData(
            self.node_merge_feature).default  # 默认feature
        self.edge_input_feature = self.feature_process_edge()  # 边——类型——数据
        self.edge_merge_feature = self.feature_merge_edge()  # 边——数据
        self.edge_defau_feature = DictData(
            self.edge_merge_feature).default  # 默认feature

    def feature_process_node(self):
        result = {}
        for node in self.basedata.nodes:
            result[node] = {}
        for feat_type in self.basedata.origin_node_embed.keys():
            data_statistic = DictData(
                self.basedata.origin_node_embed[feat_type])
            embedding = self.basedata.origin_node_embed[feat_type]
            for node in self.basedata.nodes:
                if node in embedding.keys():
                    result[node][feat_type] = embedding[node]
                else:
                    result[node][feat_type] = data_statistic.default
        return result

    def feature_merge_node(self):
        result = {}
        for node in self.node_input_feature.keys():
            merge_data = self.node_input_feature[node]['gene_expression']
            result[node] = merge_data
        return result

    def feature_process_edge(self):
        result = {}
        for edge in self.basedata.edges:
            result[edge] = {}
        for feat_type in self.basedata.origin_edge_embed.keys():
            data_statistic = DictData(
                self.basedata.origin_edge_embed[feat_type])
            embedding = self.basedata.origin_edge_embed[feat_type]
            for edge in self.basedata.edges:
                if edge in embedding.keys():
                    result[edge][feat_type] = embedding[edge]
                else:
                    result[edge][feat_type] = data_statistic.default
        return result

    def feature_merge_edge(self):
        result = {}
        for edge in self.edge_input_feature.keys():
            merge_data = self.edge_input_feature[edge]['temp']
            result[edge] = merge_data
        return result


class WholeGraphWithEmbedding():
    def __init__(self, name, nodes, edges, n_emb, e_emb, typ):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.n_emb = n_emb
        self.e_emb = e_emb
        self.typ = self.direction_repair(typ)
        self.mapping = self.get_mapping(self.nodes)
        self.base_graph = self.graph_data()

    def direction_repair(self, typ):
        if typ is False:
            new_edges = self.edges.copy()
            for edge in self.edges:
                new_edges.append(tuple(edge[::-1]))
            self.edges = new_edges
            new_e_emb = self.e_emb.copy()
            for edge in self.e_emb.keys():
                new_e_emb[tuple(edge[::-1])] = self.e_emb[edge]
            self.e_emb = new_e_emb
        return typ

    def get_mapping(self, nodes):
        result = {}
        for index, node in enumerate(nodes):
            result[node] = index
        return result

    def graph_data(self):
        nx_graph = nx.DiGraph()
        nx_graph.name = self.name
        for node in self.nodes:
            n_index = self.mapping[node]
            nx_graph.add_node(n_index, feature=self.n_emb[node])
        for edge in self.edges:
            e1_index, e2_index = self.mapping[edge[0]], self.mapping[edge[1]]
            nx_graph.add_edge(e1_index, e2_index, feature=self.e_emb[edge])
            nx_graph.add_edge(e2_index,
                              e1_index,
                              feature=self.e_emb[edge[::-1]])
        return nx_graph


class SingleGraph():
    def __init__(self, nx_graph, dgl_graph, comple, typ):
        self.nx_graph = nx_graph
        self.dgl_graph = dgl_graph
        self.comple = comple
        self.typ = typ
        self.label = None


class DGLdata():
    def __init__(self, nx_graph, mapping, bench, compare, def_nemb, def_eemb):
        self.nx_base_graph = nx_graph
        self.bench = bench
        self.compare = compare

        self.def_nemb = def_nemb  # 默认编码
        self.def_eemb = def_eemb  # 默认编码
        self.nx_expand_graph, self.mapping = self.expand_graph(bench, mapping)
        self.dgl_base_graph = self.nex_to_dgl(self.nx_expand_graph)

        self.nx_bench_graphs = self.extract_graphs(self.nx_expand_graph, bench)
        self.nx_compare_graphs = self.extract_graphs(self.nx_expand_graph,
                                                     compare)
        self.dgl_bench_graphs = [
            self.nex_to_dgl(item) for item in self.nx_bench_graphs
        ]
        self.dgl_compare_graphs = [
            self.nex_to_dgl(item) for item in self.nx_compare_graphs
        ]
        self.nx_random_graphs, self.rands = self.extract_rand_graphs(
            self.nx_expand_graph, [len(item) for item in bench], 1)
        self.dgl_random_graphs = [
            self.nex_to_dgl(item) for item in self.nx_random_graphs
        ]

        self.all_bench = self.datas_to_singlegraph(self.nx_bench_graphs,
                                                   self.dgl_bench_graphs,
                                                   self.bench, 'bench')
        self.all_compare = self.datas_to_singlegraph(self.nx_compare_graphs,
                                                     self.dgl_compare_graphs,
                                                     self.compare, 'compare')
        self.all_random = self.datas_to_singlegraph(self.nx_random_graphs,
                                                    self.dgl_random_graphs,
                                                    self.rands, 'random')

    def datas_to_singlegraph(self, nx_graphs, dgl_graphs, complexs, typ):
        assert len(nx_graphs) == len(dgl_graphs) == len(complexs)
        result = []
        for index in range(len(nx_graphs)):
            single_graph = SingleGraph(nx_graphs[index], dgl_graphs[index],
                                       complexs[index], typ)
            result.append(single_graph)
        return result

    def expand_graph(self, bench, mapping):
        expand_graph = self.nx_base_graph.copy()
        index = len(mapping.keys())
        for b in bench:
            for b_i in b:
                if b_i not in mapping.keys():
                    mapping[b_i] = index
                    expand_graph.add_node(index, feature=self.def_nemb)
                    index += 1
        return expand_graph, mapping

    def extract_graphs(self, graph, comp_list):
        result = []
        for item in comp_list:
            mapped_comp = self.mapping_res(item)
            sub_graph = nx.subgraph(graph, mapped_comp)
            result.append(sub_graph)
            # max_compo_graph = self.get_max_compo(sub_graph)
            # if len(max_compo_graph.nodes) >= 2 and (len(max_compo_graph.nodes)/len(item) > 0.5):
            #     result.append(max_compo_graph)
            # else:
            #     print("mismatch bench:", item)
        return result

    def extract_rand_graphs(self, graph: nx.Graph, lenght_list, dup):
        result_graphs = []
        result_complexs = []
        back_mapping = {}
        for item in self.mapping.keys():
            back_mapping[self.mapping[item]] = item
        for _ in range(dup):
            for lenght in lenght_list:
                success = False
                while success is False:
                    temp_graph, nodes = self.extract_rand_graph(graph, lenght)
                    if nodes is not None:
                        success = True
                        result_graphs.append(temp_graph)
                        comp = [back_mapping[item] for item in nodes]
                        result_complexs.append(comp)
        return result_graphs, result_complexs

    def extract_rand_graph(self, graph: nx.Graph, size):
        beginer = random.choice(list(graph.nodes.keys()))
        node_sets = set([beginer])
        neighbor_sets = set(graph.neighbors(beginer))
        while len(node_sets) < size:
            try:
                next_node = random.choice(list(neighbor_sets))
                neighbor_sets.remove(next_node)
                node_sets.add(next_node)
                for nei in graph.neighbors(next_node):
                    if nei not in node_sets:
                        neighbor_sets.add(nei)
            except Exception:
                return None, None
        node_list = list(node_sets)
        graph = nx.subgraph(graph, node_list)
        return graph, node_list

    def get_max_compo(self, graph):  # 暂时不保证连通性
        conn_compo = nx.connected_components(nx.Graph(graph))  # 转为无向图判断
        max_num = 0
        res_nodes = []
        for item in conn_compo:
            if len(item) > max_num:
                res_nodes = item
                max_num = len(item)
        return nx.subgraph(graph, res_nodes)

    def mapping_res(self, complexs):
        result = []
        for item in complexs:
            result.append(self.mapping[item])  # 保证mapping齐全
        return result

    def nex_to_dgl(self, graph: nx.Graph):
        dgl_graph = dgl.DGLGraph()
        temp_map = {}
        for index, node in enumerate(graph.nodes):
            feat = torch.tensor(graph.nodes[node]['feature'],
                                dtype=torch.float32).reshape(1, -1)
            dgl_graph.add_nodes(1, data={'feature': feat})
            dgl_graph.add_edge(
                index,
                index,
                data={
                    'feature':
                    torch.tensor(  # 需要添加默认数据
                        self.def_eemb, dtype=torch.float32).reshape(1, -1)
                })
            temp_map[node] = index
        for edge in graph.edges:
            v0 = temp_map[edge[0]]
            v1 = temp_map[edge[1]]
            feat = torch.tensor(graph[edge[0]][edge[1]]['feature'],
                                dtype=torch.float32).reshape(1, -1)
            dgl_graph.add_edge(v0, v1, data={'feature': feat})
        return dgl_graph


def main():
    basedata = DataLoaderGraph(
        'dip', r'D:\code\gao_complex\Data\protein\network\dip_17201',
        r'D:\code\gao_complex\Data\protein\embedding')
    featured_data = FeatureEngineer(basedata)
    bench_data = DataLoaderFiles(
        r'D:\code\gao_complex\Data\protein\bench\CYC2008_408')
    compare_data = DataLoaderFiles(
        r'D:\code\gao_complex\Data\protein\cut\dip_17201-coach')
    graphdata = WholeGraphWithEmbedding(basedata.graph_name,
                                        basedata.nodes,
                                        basedata.edges,
                                        featured_data.node_merge_feature,
                                        featured_data.edge_merge_feature,
                                        typ=False)
    dgldata = DGLdata(graphdata.base_graph, graphdata.mapping,
                      bench_data.merge_datas, compare_data.merge_datas,
                      featured_data.node_defau_feature,
                      featured_data.edge_defau_feature)
    return dgldata


if __name__ == "__main__":
    main()
    pass
