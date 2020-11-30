import pandas as pd
import random
import networkx as nx
import dgl
import torch
import pickle
import queue
import os
import numpy as np
from sklearn import preprocessing
from multiprocessing import pool as mtp
from matplotlib import pyplot as plt


def read_datas(node_path, edge_path):
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


# 数据预处理，一些归一化等等
def dataprocess(matrix):
    matrix = np.array(matrix)
    matrix = preprocessing.normalize(matrix, axis=0)
    return matrix


def get_graph(nodes, nodematrix, edges, edgematrix, direct):  # 永远当成有向图处理，无向图也需要转为有向图
    nx_graph = nx.DiGraph()
    for index, item in enumerate(nodematrix):
        nx_graph.add_node(nodes[index], w=item)
    if direct:
        for index, item in enumerate(edgematrix):
            nx_graph.add_edge(edges[index][0], edges[index][1], w=item)
    else:
        for index, item in enumerate(edgematrix):  # 无向图可以这么处理，重复
            nx_graph.add_edge(edges[index][0], edges[index][1], w=item)
            nx_graph.add_edge(edges[index][1], edges[index][0], w=item)
    return nx_graph


def subgraphs(complexes, graph):
    res = []
    for comp in complexes:
        subgraph = nx.subgraph(graph, comp)
        subgraph_bi = nx.Graph(subgraph)  # 转换为有向图求解
        sub_components = nx.connected_components(subgraph_bi)
        for sub_component in sub_components:
            res.append(sub_component)
    return res


def get_random_graphs(graph, l_list, target, multi=False):
    # 好像多进程版本并没有太多效果
    res = list()
    if multi:
        pool = mtp.Pool(processes=5)
        for i in range(target):
            size = random.choice(l_list)
            res.append(pool.apply_async(
                get_single_random_graph_nodes, args=(graph, size)))
        pool.close()
        pool.join()
        return [item.get() for item in res]
    else:
        for i in range(target):
            size = random.choice(l_list)
            res.append(get_single_random_graph_nodes(graph, size))
        return res


def get_single_random_graph_nodes(graph, size):  # 这种随机化结果产生的区分度过强，看有没有其他随机的方案
    # 注意随机游走的时候将有向图当成无向图处理
    all_nodes = list(graph.nodes.keys())  # 按照权重取值
    all_node_weights = [graph.degree(node) for node in all_nodes]
    beginer = random.choices(all_nodes, weights=all_node_weights, k=1)[0]
    # 按照权重选取下一个点
    node_set = set([beginer])
    neighbor_lists = list(set(graph.successors(beginer)) | set(
        graph.predecessors(beginer)))  # 生成随机图的时候有向图当成无向图处理
    neighbor_weights = [1 for i in range(len(neighbor_lists))]
    max_weight = 1
    while len(node_set) < size and len(neighbor_lists):
        next_node = random.choices(
            neighbor_lists, weights=neighbor_weights, k=1)[0]
        node_index = neighbor_lists.index(next_node)
        neighbor_lists.pop(node_index)
        the_weight = neighbor_weights.pop(node_index)
        max_weight = max(max_weight, the_weight)

        if (the_weight == 1 and max_weight >= 100) or (the_weight == 10 and max_weight >= 10000):  # 密集子图之后不应该再出现低权重图
            continue

        node_set.add(next_node)
        neis = list(set(graph.successors(beginer)) | set(
            graph.predecessors(beginer)))  # 区分有向图和无向图
        for nei in neis:
            if nei not in node_set:
                if nei in neighbor_lists:
                    nei_index = neighbor_lists.index(nei)
                    neighbor_weights[nei_index] *= 10  # 强调
                else:
                    neighbor_lists.append(nei)
                    neighbor_weights.append(1)
    sub_graph = nx.subgraph(graph, node_set)  # 最后还需要在子图里面去除1/4的度小的节点
    items = [(node, sub_graph.degree(node)) for node in node_set]
    remove_num_direct = min(len(node_set)//4, 6)  # 也不能去除太多了，最多去除6个

    degrees = [sub_graph.degree(node) for node in node_set]
    meandegree = (sum(degrees)/len(degrees))
    sitems = sorted(items, key=lambda i: i[1])
    res = set()
    for item in sitems[remove_num_direct:]:
        if item[1] > int(meandegree/2):  # 按照平均度数再减去一部分
            res.add(item[0])
    return res


def read_bench(path):
    res = list()
    with open(path, 'r')as f:
        for line in f:
            line_splited = None
            if '\t' in line:
                line_splited = line.strip().split('\t')
            elif ' ' in line:
                line_splited = line.strip().split(' ')
            else:
                pass
            res.append(set(line_splited))
    return res


# 具体怎么做以后需要改进，子图合并操
def merged_data(items):
    all_merged_res = []
    for item in items:
        cur_merge_target = []
        tempres = item
        for index, single_res in enumerate(all_merged_res):
            if len(item & single_res)/(len(item | single_res)) > 0.8:
                cur_merge_target.append(index)
                tempres = tempres | single_res
        for removeindex in cur_merge_target[::-1]:
            all_merged_res.pop(removeindex)  # 从后面往前面剔除
        all_merged_res.append(tempres)
    res = list()
    for data in all_merged_res:
        if len(data) >= 2:
            res.append(data)
    return res


# 去重处理
def remove_duplicate(complexes, targets):
    res = []
    for comp in complexes:
        dup = False
        for targ in targets:
            if len(comp & targ)/(len(comp | targ)) > 0.8:
                dup = True
                break
        if dup is False:
            res.append(comp)
    return res


def showsubgraphs(graph, nodelists, path):
    os.makedirs(path, exist_ok=True)
    for index, nodes in enumerate(nodelists):
        subgraph = nx.subgraph(graph, nodes)
        nx.draw(subgraph)
        plt.savefig(path+"/{}".format(index))
        plt.close()


def get_singlegraph(biggraph, nodes, direct, label, index):
    print('processing {}'.format(index))
    subgraph = biggraph.subgraph(nodes)
    return single_data(subgraph, direct, label)


class single_data:
    def __init__(self, graph, direct, label=None):
        self.label = label
        self.graph = self.dgl_graph(graph)
        self.feat = torch.tensor(self.get_default_feature(
            graph, direct), dtype=torch.float32).reshape(1, -1)

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
            res.add_edges(nodes.index(v0), nodes.index(v1), {'feat': data})
        return res

    def get_default_feature(self, graph: nx.Graph, direct):
        result = []
        result.append(len(graph.nodes))
        result.append(nx.density(graph))
        degrees = nx.degree(graph)
        degrees = np.array([item[1]for item in degrees])
        clusters = nx.clustering(graph)
        clusters = np.array([clusters[item] for item in clusters.keys()])
        # topologic = nx.topological_sort(graph)
        result.append(degrees.mean())
        result.append(degrees.max())
        result.append(degrees.min())
        result.append(degrees.var())

        result.append(clusters.mean())
        result.append(clusters.max())
        result.append(clusters.var())
        if direct:
            # 计算有方向的时候的补充特征
            pass
        else:
            # 计算无向的时候的补充特征
            correlation = nx.degree_pearson_correlation_coefficient(
                graph)
            result.append(correlation if correlation is not np.nan else 0.0)
        return list(result)


def first_stage(node_path, edge_path, postive_path, middle_path, save_path, reload=True, direct=False):
    if not reload and os.path.exists(save_path):
        with open(save_path, 'rb')as f:
            result = pickle.load(f)
        return result
    '''
    下面是读取点数据，和边数据，并做特征初始化处理
    '''
    # 获取图数据
    nodes, nodematrix, edges, edgematrix = read_datas(node_path, edge_path)
    # 归一化处理
    nodematrix = dataprocess(nodematrix)
    edgematrix = dataprocess(edgematrix)
    nx_graph = get_graph(nodes, nodematrix, edges, edgematrix, direct)
    # dgl_graph = single_data(nx_graph, direct).graph
    bench_data = read_bench(postive_path)
    middle_data = read_bench(middle_path)
    random_target = (len(bench_data)+len(middle_data))  # 先多取一些，再截取需要的部分
    random_data = get_random_graphs(
        nx_graph, [len(item) for item in bench_data + middle_data], random_target)  # TODO 设定随机的数目

    # 接下来需要提取真正的graph，找出所有的subgraph
    bench_data = subgraphs(bench_data, nx_graph)
    middle_data = subgraphs(middle_data, nx_graph)
    # 接下来归并处理
    bench_data = merged_data(bench_data)  # 621->555
    middle_data = merged_data(middle_data)  # 888->416
    random_data = merged_data(random_data)  # 129->99
    # 接下来去重
    middle_data = remove_duplicate(middle_data, bench_data)[:len(bench_data)]
    random_data = remove_duplicate(
        random_data, bench_data+middle_data)[:len(bench_data)]
    # 存储图片
    # showsubgraphs(nx_graph, bench_data, "Data/Yeast/pictures/bench")
    # showsubgraphs(nx_graph, middle_data, "Data/Yeast/pictures/middle")
    # showsubgraphs(nx_graph, random_data, "Data/Yeast/pictures/random")
    # 整理成数据集
    all_datas = []
    all_datas.extend([item, 0] for item in bench_data)
    all_datas.extend([item, 1] for item in middle_data)
    all_datas.extend([item, 2] for item in random_data)
    # 多进程处理
    multi_res = []
    pool = mtp.Pool(processes=10)
    for index, item in enumerate(all_datas):  # TODO 目前调试流程只选取500个
        multi_res.append(pool.apply_async(
            get_singlegraph, args=(nx_graph, item, direct, -1, index)))
    pool.close()
    pool.join()
    datasets = [item.get() for item in multi_res]

    with open(save_path, 'wb') as f:
        pickle.dump(datasets, f)
    return datasets


def second_stage(node_path, edge_path, candi_path, save_path, reload=True, direct=False):
    if not reload and os.path.exists(save_path):
        with open(save_path, 'rb')as f:
            result = pickle.load(f)
        return result
    # 获取图数据
    nodes, nodematrix, edges, edgematrix = read_datas(node_path, edge_path)
    # 归一化处理
    nodematrix = dataprocess(nodematrix)
    edgematrix = dataprocess(edgematrix)
    nx_graph = get_graph(nodes, nodematrix, edges, edgematrix, direct)
    candi_data = read_bench(candi_path)
    # datasets = [get_singlegraph(nx_graph, item, direct, -1)
    #             for item in candi_data]  # -1代表无意义
    # TODO 注意那就不需要考虑不连通的情况，因为这是在我给定的图里面获取的
    # return datasets
    multi_res = []
    pool = mtp.Pool(processes=10)
    for index, item in enumerate(candi_data[:500]):  # TODO 目前调试流程只选取500个
        multi_res.append(pool.apply_async(
            get_singlegraph, args=(nx_graph, item, direct, -1, index)))
    pool.close()
    pool.join()
    datasets = [item.get() for item in multi_res]
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
