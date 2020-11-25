import pandas as pd
import random
import networkx as nx
import dgl
import torch
import pickle
import os
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt


def read_graph(node_path, edge_path):
    nodes, nodematrix = [], []
    edges, edgematrix = [], []

    with open(node_path, 'r') as f:
        next(f)
        for nodedata in f:
            nodedata_splited = nodedata.split('\t')
            node_id = nodedata_splited[0]
            node_feat = list(map(float, nodedata_splited[1:]))
            nodes.append(node_id)
            nodematrix.append(node_feat)

    with open(edge_path, 'r')as f:
        next(f)
        for edgedata in f:
            edgedata_splited = edgedata.split('\t')
            v0, v1 = edgedata_splited[0].split(' ')
            edge_feat = list(map(float, edgedata_splited[1:]))
            edges.append([v0, v1])
            edgematrix.append(edge_feat)
    return nodes, nodematrix, edges, edgematrix


def dataprocess(matrix):
    matrix = np.array(matrix)
    matrix = preprocessing.normalize(matrix, axis=0)
    return matrix


def get_graph(nodes, nodematrix, edges, edgematrix):
    res = nx.Graph()
    for index, item in enumerate(nodematrix):
        res.add_node(nodes[index], w=item)
    for index, item in enumerate(edgematrix):
        res.add_edge(edges[index][0], edges[index][1], w=item)
    return res


def get_random_graphs(graph, l_list, target):
    result = set()
    while len(result) < target:
        size = random.choice(l_list)
        success = False
        while success is False:
            nodes = get_single_random_graph_nodes(graph, size)
            if nodes is not None and nodes not in result:
                success = True
                result.add(nodes)
    return result


def get_single_random_graph_nodes(graph, size):  # 这种随机化结果产生的区分度过强，看有没有其他随机的方案
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


def remove_fake_graph(datas, graph):
    res = set()
    nodes = set(graph.nodes)
    for data in datas:
        if len(set(data)-nodes) != 0:
            continue
        subgraph = graph.subgraph(data)
        '''
        #如果具有多余一个连通子图，则跳过
        sub_compos = nx.connected_components(subgraph)
        bigest_graph = next(sub_compos)
        if len(bigest_graph) != len(subgraph.nodes):
            continue
        '''
        '''
        # 如果具有一个点的邻居数为0，则跳过
        miniest = 1
        for node in subgraph.nodes:
            miniest = min(miniest, nx.degree(subgraph, node))
        if miniest == 0:
            continue
        '''

        # 只有一个图有多个部分组成，而且最大的部分小于原来的80%的时候才跳过
        sub_compos = nx.connected_components(subgraph)
        bigest_graph = next(sub_compos)
        if len(bigest_graph) < int(len(subgraph.nodes)*0.70):
            continue

        res.add(data)
    return res


def showsubgraphs(graph, nodelists, path):
    os.makedirs(path, exist_ok=True)
    for index, nodes in enumerate(nodelists):
        subgraph = nx.subgraph(graph, nodes)
        nx.draw(subgraph)
        plt.savefig(path+"/{}".format(index))
        plt.close()


class single_data:
    def __init__(self, graph, label):
        self.label = label
        self.graph = self.dgl_graph(graph)
        self.feat = torch.tensor(self.get_default_feature(
            graph), dtype=torch.float32).reshape(1, -1)

    def dgl_graph(self, graph: nx.Graph):
        res = dgl.DGLGraph()
        nodes = list(graph.nodes)
        for index, node in enumerate(nodes):
            data = torch.tensor(
                graph.nodes[node]['w'], dtype=torch.float32).reshape(1, -1)
            deg = torch.tensor(graph.degree(
                node), dtype=torch.float32).reshape(1, -1)
            res.add_nodes(1, {'feat': data, 'degree': deg})
            # res.add_edge(index, index)
        for v0, v1 in graph.edges:
            data = torch.tensor(
                graph[v0][v1]['w'], dtype=torch.float32).reshape(1, -1)
            res.add_edge(nodes.index(v0), nodes.index(v1), {'feat': data})
            res.add_edge(nodes.index(v1), nodes.index(v0), {'feat': data})
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
            graph)

        result.append(degrees.mean())
        result.append(degrees.max())
        result.append(degrees.min())
        result.append(degrees.var())

        result.append(clusters.mean())
        result.append(clusters.max())
        result.append(clusters.var())

        result.append(correlation if correlation is not np.nan else 0.0)
        return list(result)


def first_stage(reload=False):
    node_path = "Data/Yeast/embedding/dip_node"
    edge_path = "Data/Yeast/embedding/dip_edge"
    postive_path = "Data/Yeast/bench/CYC2008"
    middle_path = "Data/Yeast/bench/dip_coach"
    save_path = "Data/Yeast/first_stage"
    if not reload:
        with open(save_path, 'rb')as f:
            result = pickle.load(f)
        return result
    nodes, nodematrix, edges, edgematrix = read_graph(node_path, edge_path)
    nodematrix = dataprocess(nodematrix)
    edgematrix = dataprocess(edgematrix)
    graph = get_graph(nodes, nodematrix, edges, edgematrix)

    get_single_random_graph_nodes(graph, 5)
    bench_data = read_bench(postive_path)
    middle_data = read_bench(middle_path)

    bench_data_remove_small = remove_small_graph(bench_data, 3)  # 236个
    middle_data_remove_small = remove_small_graph(middle_data, 3)  # 883

    bench_data_remove_fake_graph = remove_fake_graph(
        bench_data_remove_small, graph)
    # 142 去除个数为2的，保证为全连通图，对于cyc2008数据集来说，只剩下142个子图
    # 143 不应该这么严格，只需要去除孤立的点，那么可以剩下143个子图
    middle_data_remove_fake_graph = remove_fake_graph(
        middle_data_remove_small, graph)  # 882
    middle_data_remove_fake_graph = set(
        list(middle_data_remove_fake_graph)[:180])

    random_data_size_list = [len(item)
                             for item in bench_data_remove_fake_graph]
    random_target = 180
    random_data = get_random_graphs(
        graph, random_data_size_list, random_target)
    statics_nodes = {
        'pos': bench_data_remove_fake_graph,
        'mid': middle_data_remove_fake_graph,
        'neg': random_data
    }

    showsubgraphs(graph, bench_data_remove_fake_graph, "Data/Yeast/temp/bench")
    showsubgraphs(graph, middle_data_remove_fake_graph,
                  "Data/Yeast/temp/middle")
    showsubgraphs(graph, random_data, "Data/Yeast/temp/random")

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
    return datasets


class BatchGenerator():
    def __init__(self, data, batch_size):
        self.data = data
        random.shuffle(self.data)
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


if __name__ == "__main__":
    pass