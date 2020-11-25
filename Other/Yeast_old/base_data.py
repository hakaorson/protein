import os
import re
import dgl
from dgl import subgraph
import numpy as np
import networkx as nx
import torch
import random
import math
import pickle


def read_file(path):  # 读取文件的常规方法
    result = []
    with open(path, 'r') as f:
        for line in f:
            data = re.split(' |\t', line.strip())
            result.append(data)
    return result


def edge_to_nxgraph(edges):
    graph = nx.Graph()
    for v0, v1 in edges:
        graph.add_edge(v0, v1)
    return graph


class BaseDatas():
    def __init__(self,
                 path,
                 network_name,
                 bench_name=None,
                 compare_name=None,
                 node_embedding_names=None,
                 edge_embedding_names=None):
        self.path = path
        self.category = os.path.basename(self.path)
        self.graph_path = os.path.join(self.path, 'network', network_name)
        self.edges = self.read_tuple(self.graph_path)
        self.nodes = self.get_itemset(self.edges)
        self.node_embed_names = node_embedding_names
        self.edge_embed_names = edge_embedding_names

        self._bench_path = os.path.join(self.path, 'bench', bench_name)
        self.bench_data = self.read_tuple(self._bench_path)
        self._compare_path = os.path.join(self.path, 'compare', compare_name)
        self.compare_data = self.read_tuple(self._compare_path)

        self._edge_embed_path = os.path.join(self.path, 'embedding', 'edge')
        self.edge_embed_data = self.read_embedding(
            self._edge_embed_path, self.edge_embed_names, key_size=2)
        self._node_embed_path = os.path.join(self.path, 'embedding', 'node')
        self.node_embed_data = self.read_embedding(
            self._node_embed_path, self.node_embed_names, key_size=1)
        pass

    def get_itemset(self, datas):  # 获取数据里面的所有元素
        itemset = set()
        for line in datas:  # 每一行
            for item in line:  # 每一个
                itemset.add(item)
        return itemset

    def read_tuple(self, path):
        data = read_file(path)
        result = set()
        for item in data:
            result.add(tuple(item))
        return result

    def read_dict(self, path, key_size=1):
        data = read_file(path)
        result = dict()
        if key_size == 1:
            for item in data:
                feat_data = list(map(float, item[key_size:]))
                result[item[0]] = feat_data
        else:
            for item in data:
                feat_data = list(map(float, item[key_size:]))
                result[tuple(item[:key_size])] = feat_data
        return result

    def read_embedding(self, path, names, key_size=1):
        result = {}
        for embed_name in names:
            embed_path = os.path.join(path, embed_name)
            embed_data = self.read_dict(embed_path, key_size)
            result[embed_name] = embed_data
        return result


class ProcessedData(BaseDatas):
    def __init__(self, base_data: BaseDatas):
        self._base_data = base_data
        self.category = self._base_data.category
        self.all_nodes = self.expand_nodes(
            self._base_data.nodes, self._base_data.bench_data)  # 所有需要具有feature的节点
        self.all_edges = self.expand_edges(
            self._base_data.edges, self.all_nodes)  # 准备好所有的边
        self._nx_base = edge_to_nxgraph(self._base_data.edges)
        self.random_data = self.get_random_graphs(
            self._nx_base, [len(item)for item in self._base_data.bench_data])  # 随机

        self.set_default_embed_edge()
        self.set_default_embed_node()

        self.node_embed_map = self.get_embedding_map(
            self.all_nodes, self._base_data.node_embed_data)
        self.edge_embed_map = self.get_embedding_map(
            self.all_edges, self._base_data.edge_embed_data)

    def expand_nodes(self, nodes, bench_data):
        bench_data_nodes = self.get_itemset(bench_data)
        nodes = nodes | bench_data_nodes
        return nodes

    def expand_edges(self, edges, nodes):
        result = set()
        for edge in edges:
            result.add(edge)
            result.add(tuple(edge[::-1]))
        for node in nodes:
            result.add(tuple([node, node]))
        return result

    def get_feature_arrays(self, features):
        result = {}
        for feat in features:
            data_list = []
            for item in features[feat].keys():
                data_list.append(features[feat][item])
            data_array = np.array(data_list)
            result[feat] = data_array
        return result

    def set_default_embed_edge(self):
        feat_arrays = self.get_feature_arrays(self._base_data.edge_embed_data)
        for typ in self._base_data.node_embed_names:
            if typ == 'base':
                default_data = list(np.mean(feat_arrays[typ], 0))
                loop_data = [2.0]
            else:
                raise KeyError
            self._base_data.edge_embed_data[typ][('DEFAULT',)] = default_data
            self._base_data.edge_embed_data[typ][('LOOP',)] = loop_data

    def set_default_embed_node(self):
        feat_arrays = self.get_feature_arrays(self._base_data.node_embed_data)
        for typ in self._base_data.node_embed_names:
            if typ == 'gene_expression':
                default_data = list(np.mean(feat_arrays[typ], 0))
            elif typ == 'base':
                default_data = [1.0]
            else:
                raise KeyError
            self._base_data.node_embed_data[typ][('DEFAULT',)] = default_data

    def get_random_graphs(self, graph, l_list):
        result = set()
        for length in l_list:
            success = False
            while success is False:
                nodes = self.random_single_graph(graph, length)
                if nodes is not None and nodes not in result:
                    success = True
                    result.add(nodes)
        return result

    def random_single_graph(self, graph: nx.Graph, size):
        beginer = random.choice(list(graph.nodes.keys()))
        node_set = set([beginer])
        neighbor_sets = set(graph.neighbors(beginer))
        while len(node_set) < size:
            try:
                next_node = random.choice(list(neighbor_sets))
                neighbor_sets.remove(next_node)
                node_set.add(next_node)
                for nei in graph.neighbors(next_node):
                    if nei not in node_set:
                        neighbor_sets.add(nei)
            except Exception:
                return None
        return tuple(node_set)

    def get_embedding_map(self, keys, feat_map):
        result = {}
        for key in keys:
            feature = []
            for feat in feat_map.keys():
                if key not in feat_map[feat].keys():
                    if tuple(key[::-1]) in feat_map[feat].keys():
                        feature.extend(feat_map[feat][tuple(key[::-1])])
                    if len(list(key)) == 2 and list(key[0]) == list(key[1]):
                        target = ('LOOP',)
                    else:
                        target = ('DEFAULT',)
                    feature.extend(feat_map[feat][target])
                else:
                    feature.extend(feat_map[feat][key])
            result[key] = feature
        return result


class NXGraph():
    def __init__(self, nodes, edges, n_embedding, e_embedding):
        self.base_graph = self.get_basegraph(nodes, edges)
        self.feat_graph = self.get_featgraph(
            self.base_graph, n_embedding, e_embedding)

    def get_basegraph(self, nodes, edges):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)
        return nx_graph

    def get_featgraph(self, graph, n_embed, e_embed):
        for node in graph.nodes:
            graph.nodes[node]['feature'] = n_embed[node]
        for v0, v1 in graph.edges:
            graph[v0][v1]['feature'] = e_embed[tuple([v0, v1])]
        return graph

    def sub_graphs(self, sub_ids):
        result = []
        for item in sub_ids:
            temp_graph = nx.subgraph(self.feat_graph, item)
            result.append(temp_graph)
        return result


class DGLGraph():
    def __init__(self, all_nodes, all_edges, node_embedding=None, edge_embedding=None):
        self.nodes = all_nodes
        self.edges = all_edges
        self.node_embed = node_embedding
        self.edge_embed = edge_embedding
        self.map_idindex, self.map_indexid = self.mapping(self.nodes)
        self.dgl_graph = self.get_dgl_graph(self.nodes, self.edges)

    def mapping(self, nodes):
        id_index, index_id = {}, {}
        for index, node in enumerate(nodes):
            id_index[node] = index
            index_id[index] = node
        return id_index, index_id

    def get_dgl_graph(self, nodes, edges):
        graph = dgl.DGLGraph()
        for node in self.nodes:
            data = torch.tensor(
                self.node_embed[node], dtype=torch.float32).reshape(1, -1)
            graph.add_nodes(1, {'feature': data})
        for v0, v1 in self.edges:
            data = torch.tensor(
                self.edge_embed[(v0, v1)], dtype=torch.float32).reshape(1, -1)
            v0_index, v1_index = self.map_idindex[v0], self.map_idindex[v1]
            graph.add_edge(v0_index, v1_index, {'feature': data})
        return graph


class SingleGraph():
    def __init__(self, nx_graph, label):
        self.nx_graph = nx_graph
        self.base_feature = self.get_default_feature(self.nx_graph)
        self.base_feature = torch.tensor(
            self.base_feature, dtype=torch.float32).reshape(1, -1)
        self.nodes = nx_graph.nodes
        self.dgl_graph = self.get_dgl_graph(self.nx_graph)
        self.label = label

    def get_default_feature(self, graph: nx.Graph):
        result = []
        result.append(len(graph.nodes))
        result.append(nx.density(graph))
        degrees = nx.degree(graph)
        degrees = np.array([item[1]for item in degrees])
        clusters = nx.clustering(graph)
        clusters = np.array([clusters[item] for item in clusters.keys()])
        # topologic = nx.topological_sort(graph)
        correlation = nx.degree_pearson_correlation_coefficient(graph)

        result.append(degrees.mean())
        result.append(degrees.max())
        result.append(degrees.min())
        result.append(degrees.var())

        result.append(clusters.mean())
        result.append(clusters.max())
        result.append(clusters.var())

        result.append(correlation if correlation is not np.nan else 0.0)
        return list(result)

    def get_dgl_graph(self, nx_graph: nx.classes.graph.Graph):
        graph = dgl.DGLGraph()
        temp_map = {}
        for index, node in enumerate(nx_graph.nodes):
            data = torch.tensor(
                nx_graph.nodes[node]['feature'], dtype=torch.float32).reshape(1, -1)
            deg = torch.tensor(nx_graph.degree(
                node), dtype=torch.float32).reshape(1, -1)
            graph.add_nodes(1, {'feature': data, 'degree': deg})
            temp_map[node] = index
        for v0, v1 in nx_graph.edges:
            data = torch.tensor(
                nx_graph[v0][v1]['feature'], dtype=torch.float32).reshape(1, -1)
            v0_index, v1_index = temp_map[v0], temp_map[v1]
            if v0_index != v1_index:
                graph.add_edge(v0_index, v1_index, {'feature': data})
                graph.add_edge(v1_index, v0_index, {'feature': data})
            else:
                graph.add_edge(v0_index, v1_index, {'feature': data})
        return graph


class BatchGenerator():
    def __init__(self, data, batch_size):
        self.data = data
        # random.shuffle(self.data)
        self.batch_size = batch_size if batch_size != -1 else len(self.data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.data[self.index+self.batch_size-1]  # 用于检查是否越界
            b_data = self.data[self.index:self.index+self.batch_size]
        except IndexError:
            raise StopIteration()
        self.index += self.batch_size
        return b_data


def save_pickle(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb')as f:
        result = pickle.load(f)
    return result


def main(path=r'D:\code\gao_complex\Data\protein\storation\single_graphs', recompute=False):
    if recompute is False:
        return load_pickle(path)
    base_datas = BaseDatas(r'D:\code\gao_complex\Data\protein',
                           network_name='dip_17201',
                           bench_name='CYC2008_408',
                           compare_name='coach',
                           node_embedding_names=['base'],
                           edge_embedding_names=['base'])
    expand_data = ProcessedData(base_datas)
    base_nxgraph = NXGraph(expand_data.all_nodes, expand_data.all_edges,
                           expand_data.node_embed_map, expand_data.edge_embed_map)
    bench_nxgraphs = base_nxgraph.sub_graphs(base_datas.bench_data)
    compare_nxgraphs = base_nxgraph.sub_graphs(base_datas.compare_data)
    random_nxgraphs = base_nxgraph.sub_graphs(expand_data.random_data)
    datasets = []
    datasets.extend([SingleGraph(item, 2)for item in bench_nxgraphs])
    datasets.extend([SingleGraph(item, 1)for item in compare_nxgraphs])
    datasets.extend([SingleGraph(item, 0)for item in random_nxgraphs])
    # base_dgl = DGLGraph(expand_data.all_nodes, expand_data.all_edges,
    #                     expand_data.node_embed_map, expand_data.edge_embed_map)
    save_pickle(datasets, path)
    return datasets


if __name__ == "__main__":
    main()
