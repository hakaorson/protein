import pandas as pd
import random
import networkx as nx
import dgl
import torch
import pickle
import numpy as np


def read_graph(node_path, edge_path):
    res = nx.Graph()
    with open(node_path, 'r') as f:
        for nodedata in f:
            nodedata_splited = nodedata.split('\t')
            node_id = nodedata_splited[0]
            node_feat = list(map(float, nodedata_splited[1:]))
            res.add_node(node_id, w=node_feat)
    with open(edge_path, 'r')as f:
        for edgedata in f:
            edgedata_splited = edgedata.split('\t')
            v0, v1 = edgedata_splited[:2]
            edge_feat = list(map(float, edgedata_splited[2:]))
            res.add_edge(v0, v1, w=edge_feat)
    return res


def get_random_graphs(graph, l_list, target):
    result = set()
    while len(result) < target:
        length = random.choice(l_list)
        success = False
        while success is False:
            nodes = get_single_random_graph_nodes(graph, length)
            if nodes is not None and nodes not in result:
                success = True
                result.add(nodes)
    return result


def get_single_random_graph_nodes(graph, size):
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


def read_bench(path):
    res = set()
    with open(path, 'r')as f:
        for line in f:
            line_splited = None
            if '\t' in line:
                line_splited = line.strip().split('\t')
            elif ' ' in line:
                line_splited = line.strip().split(' ')
            else:
                pass
            res.add(tuple(line_splited))
    return res


# 只留下大于等于cutnum的项
def remove_small_graph(datas, cut_num):
    res = set()
    for data in datas:
        if len(data) >= cut_num:
            res.add(data)
    return res


def remove_notingraph(datas, graph):
    res = set()
    nodes = set(graph.nodes)
    for data in datas:
        if len(set(data)-nodes) == 0:
            res.add(data)
    return res


class single_data:
    def __init__(self, graph, label):
        self.label = label
        self.graph = self.dgl_graph(graph)
        self.feat = self.get_default_feature(graph)

    def dgl_graph(self, graph: nx.Graph):
        res = dgl.DGLGraph()
        nodes = list(graph.nodes)
        for node in nodes:
            data = torch.tensor(
                graph.nodes[node]['w'], dtype=torch.float32).reshape(1, -1)
            res.add_nodes(1, {'feat': data})
        for v0, v1 in graph.edges:
            data = torch.tensor(
                graph[v0][v1]['w'], dtype=torch.float32).reshape(1, -1)
            res.add_edge(nodes.index(v0), nodes.index(v1), {'feat': data})
        return res

    def get_default_feature(self, graph: nx.Graph):
        result = []
        result.append(len(graph.nodes))
        result.append(nx.density(graph))
        degrees = nx.degree(graph)
        degrees = np.array([item[1]for item in degrees])
        clusters = nx.clustering(graph)
        clusters = np.array([clusters[item] for item in clusters.keys()])
        # topologic = nx.topological_sort(graph)
        correlation = nx.degree_pearson_correlation_coefficient(
            graph) if max(degrees) else 0

        result.append(degrees.mean())
        result.append(degrees.max())
        result.append(degrees.min())
        result.append(degrees.var())

        result.append(clusters.mean())
        result.append(clusters.max())
        result.append(clusters.var())

        result.append(correlation if correlation is not np.nan else 0.0)
        return list(result)


def load_pickle(path):
    with open(path, 'rb')as f:
        result = pickle.load(f)
    return result


if __name__ == "__main__":
    node_path = "Yeast/embedding/dip_node"
    edge_path = "Yeast/embedding/dip_edge"
    postive_path = "Yeast/bench/CYC2008"
    middle_path = "Yeast/bench/dip_coach"
    save_path = "Yeast/data"
    graph = read_graph(node_path, edge_path)
    get_single_random_graph_nodes(graph, 5)
    bench_data = read_bench(postive_path)
    middle_data = read_bench(middle_path)

    bench_data_remove_small = remove_small_graph(bench_data, 3)  # 236个
    middle_data_remove_small = remove_small_graph(middle_data, 3)  # 883

    bench_data_remove_notin_graph = remove_notingraph(
        bench_data_remove_small, graph)  # 198个

    random_data_size_list = [len(item) for item in (
        middle_data_remove_small | bench_data_remove_notin_graph)]
    random_target = 1000
    random_data = get_random_graphs(
        graph, random_data_size_list, random_target)
    statics_nodes = {
        'pos': bench_data_remove_notin_graph,
        'mid': middle_data_remove_small,
        'neg': random_data
    }
    dgl_graphs = {
        'pos': [single_data(nx.subgraph(graph, item), 0) for item in statics_nodes['pos']],
        'mid': [single_data(nx.subgraph(graph, item), 1) for item in statics_nodes['mid']],
        'neg': [single_data(nx.subgraph(graph, item), 2) for item in statics_nodes['neg']],
    }

    datasets = []
    for key in dgl_graphs.keys():
        datasets.extend(dgl_graphs[key])

    with open(save_path, 'wb') as f:
        pickle.dump(datasets, f)
    pass
